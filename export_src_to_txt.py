import os

def export_src_to_txt(target_folder="src", output_folder="gemini_export"):
    # Prüfen, ob der src-Ordner überhaupt existiert
    if not os.path.exists(target_folder):
        print(f"Fehler: Der Ordner '{target_folder}' wurde im aktuellen Verzeichnis nicht gefunden.")
        return

    os.makedirs(output_folder, exist_ok=True)
    exported_count = 0

    for dirpath, dirnames, filenames in os.walk(target_folder):
        # Ignoriere Caches und versteckte Ordner
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']

        for file in filenames:
            # Nur Python- und Markdown-Dateien
            if not (file.endswith('.py') or file.endswith('.md')):
                continue

            full_path = os.path.join(dirpath, file)
            # Relativen Pfad vom Hauptordner aus berechnen
            rel_path = os.path.relpath(full_path, ".").replace('\\', '/')
            
            # Dateinamen für den Export flach formatieren (z.B. src__models__feature_unet.py.txt)
            flat_filename = rel_path.replace('/', '__') + '.txt'
            target_path = os.path.join(output_folder, flat_filename)

            try:
                with open(full_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    
                with open(target_path, 'w', encoding='utf-8') as outfile:
                    outfile.write(f"ORIGINAL_PATH: {rel_path}\n")
                    outfile.write("=" * 50 + "\n\n")
                    outfile.write(content)
                
                exported_count += 1
            except Exception as e:
                print(f"Fehler beim Lesen/Schreiben von {rel_path}: {e}")

    print(f"Export erfolgreich beendet. {exported_count} Dateien liegen als .txt in '{output_folder}/'.")

if __name__ == "__main__":
    export_src_to_txt()