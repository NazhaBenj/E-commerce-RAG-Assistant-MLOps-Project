import csv

from src.ingestion.html_cleaner import clean_html
import src.ingestion.process_corpus as process_module


def test_clean_html_removes_scripts_and_styles(tmp_path):
    html_file = tmp_path / "page.html"

    html_file.write_text(
        """
        <html>
            <head>
                <title>Jumia</title>
                <style>.hidden { display: none; }</style>
            </head>
            <body>
                <header>Header technique</header>
                <nav>Navigation</nav>
                <main>
                    <h1>Annulation de commande</h1>
                    <p>Le client peut annuler sa commande.</p>
                </main>
                <script>alert("test")</script>
                <footer>Footer technique</footer>
            </body>
        </html>
        """,
        encoding="utf-8",
    )

    result = clean_html(str(html_file))

    assert "Annulation de commande" in result
    assert "Le client peut annuler sa commande." in result
    assert "alert" not in result
    assert "Navigation" not in result
    assert "Footer technique" not in result
    assert "Header technique" not in result


def test_clean_html_removes_hidden_elements(tmp_path):
    html_file = tmp_path / "hidden.html"

    html_file.write_text(
        """
        <html>
            <body>
                <p>Texte visible</p>
                <p hidden>Texte caché</p>
                <div aria-hidden="true">Texte invisible</div>
                <span style="display:none">Texte masqué</span>
            </body>
        </html>
        """,
        encoding="utf-8",
    )

    result = clean_html(str(html_file))

    assert "Texte visible" in result
    assert "Texte caché" not in result
    assert "Texte invisible" not in result
    assert "Texte masqué" not in result


def test_process_corpus_creates_processed_files(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed" / "source_text"
    raw_dir.mkdir()

    html_file = raw_dir / "jumia_order.html"

    html_file.write_text(
        """
        <html>
            <body>
                <h1>Suivi de commande</h1>
                <p>Consultez votre compte client.</p>
            </body>
        </html>
        """,
        encoding="utf-8",
    )

    manifest = tmp_path / "source_manifest.csv"

    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["document_id", "filename"],
        )
        writer.writeheader()
        writer.writerow({
            "document_id": "JUMIA-TEST-001",
            "filename": "jumia_order.html",
        })

    monkeypatch.setattr(process_module, "MANIFEST", manifest)
    monkeypatch.setattr(process_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(process_module, "OUTPUT_DIR", output_dir)

    process_module.process_corpus()

    generated_file = output_dir / "JUMIA-TEST-001.txt"

    assert generated_file.is_file()

    content = generated_file.read_text(encoding="utf-8")

    assert "Suivi de commande" in content
    assert "Consultez votre compte client." in content


def test_process_corpus_skips_missing_source(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()

    manifest = tmp_path / "source_manifest.csv"

    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["document_id", "filename"],
        )
        writer.writeheader()
        writer.writerow({
            "document_id": "JUMIA-MISSING",
            "filename": "missing.html",
        })

    monkeypatch.setattr(process_module, "MANIFEST", manifest)
    monkeypatch.setattr(process_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(process_module, "OUTPUT_DIR", output_dir)

    process_module.process_corpus()

    assert not (output_dir / "JUMIA-MISSING.txt").exists()