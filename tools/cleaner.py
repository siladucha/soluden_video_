#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import subprocess
import sys

HOME = Path.home()
DRY_RUN = '--dry-run' in sys.argv

def print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")

def du(path):
    """Примерный размер в ГБ"""
    try:
        total = sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())
        return round(total / (1024**3), 2)
    except Exception:
        return 0

def delete(path):
    path = Path(path).expanduser()
    if not path.exists():
        print(f"Не существует: {path}")
        return
    size = du(path)
    if DRY_RUN:
        print(f"[DRY-RUN] Удалил бы {path} (~{size} GB)")
    else:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"Удалено: {path} (~{size} GB)")
        except Exception as e:
            print(f"Ошибка при удалении {path}: {e}")

def run_cmd(cmd):
    if DRY_RUN:
        print(f"[DRY-RUN] Выполнил бы: {cmd}")
        return
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout.strip())
    except Exception as e:
        print(f"Ошибка команды {cmd}: {e}")

def main():
    print("Скрипт очистки места на macOS (Apple Silicon M4)")
    print("Режим: " + ("DRY-RUN (только показ)" if DRY_RUN else "РЕАЛЬНОЕ УДАЛЕНИЕ"))
    print("\nСначала проверь, сколько места занято:")
    run_cmd("df -h /")

    # 1. Кэш pip (модели torch/ultralytics/transformers могут весить гигабайты)
    print_section("Pip cache")
    delete("~/.cache/pip")

    # 2. Кэш Homebrew
    print_section("Homebrew cache")
    run_cmd("brew cleanup --prune=all")  # безопасно, удаляет старые версии и скачанные файлы

    # 3. Общий пользовательский кэш
    print_section("Пользовательский кэш (~/.cache)")
    delete("~/.cache")

    # 4. Временные файлы macOS
    print_section("/private/var/folders (системный temp, часто 10+ GB)")
    delete("/private/var/folders")  # осторожно, но безопасно — macOS пересоздаст

    # 5. Кэш миниатюр
    print_section("Кэш миниатюр")
    delete("~/Library/Caches/com.apple.iconservices.store")
    delete("~/.thumbnails")

    # 6. Логи
    print_section("Логи")
    delete("~/Library/Logs")
    delete("/Library/Logs")

    # 7. Trash (корзина)
    print_section("Корзина")
    delete("~/.Trash")

    # 8. Если у тебя Xcode (iOS dev)
    print_section("Xcode DerivedData (часто 20–100 GB!)")
    delete("~/Library/Developer/Xcode/DerivedData")

    print_section("Готово!")
    print("\nПосле очистки:")
    run_cmd("df -h /")

if __name__ == "__main__":
    if not DRY_RUN:
        answer = input("\nТЫ УВЕРЕН? Это удалит файлы безвозвратно! (введи yes): ")
        if answer.lower() != 'yes':
            print("Отмена.")
            sys.exit(0)
    main()