import base64
from pathlib import Path

# Укажи путь к своему attack.py
# Например, если он лежит в папке src, то:
input_path = Path("src") / "attack_static_prompts_mut.py"
# Если файл просто в той же папке, где скрипт, то:
# input_path = Path("attack.py")

# Проверяем, существует ли файл
if not input_path.exists():
    print(f"Файл не найден: {input_path.absolute()}")
    exit(1)

# Читаем и кодируем
with open(input_path, "rb") as f:
    raw_bytes = f.read()
    encoded = base64.b64encode(raw_bytes).decode("utf-8")

# Сохраняем результат в файл attack_b64.txt
output_path = Path("attack_b64.txt")
with open(output_path, "w", encoding="utf-8") as f_out:
    f_out.write(encoded)

print(f"✅ Закодировано успешно! Длина строки: {len(encoded)} символов")
print(f"Результат сохранён в: {output_path.absolute()}")