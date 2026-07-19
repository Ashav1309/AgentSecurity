
import time
from pathlib import Path
import aicomp_sdk

from aicomp_sdk.core.env.sandbox import SandboxEnv
from aicomp_sdk.agents import build_agent
from aicomp_sdk.attacks import AttackRunConfig

from attack_static_prompts_mut import AttackAlgorithm

def main():
    # 1. Путь к фикстурам (тестовые данные)
    fixtures_path = Path(aicomp_sdk.__file__).parent / "fixtures"
    print(f"Fixtures path: {fixtures_path}")

    # 2. Создаём окружение с детерминированным агентом (без LLM)
    #    seed=123 — фиксирован для воспроизводимости
    env = SandboxEnv(
        seed=123,
        fixtures_dir=fixtures_path,
        agent=build_agent("deterministic")  # имитирует поведение, но не вызывает LLM
    )
    env.reset()

    # 3. Конфиг с бюджетом времени (для теста можно поставить 60 секунд)
    config = AttackRunConfig(time_budget_s=60)

    # 4. Создаём экземпляр твоего алгоритма
    algorithm = AttackAlgorithm(config)

    # 5. Запускаем
    start = time.time()
    candidates = algorithm.run(env, config)
    elapsed = time.time() - start

    # 6. Выводим результаты
    print(f"Найдено кандидатов: {len(candidates)}")
    print(f"Время выполнения: {elapsed:.2f} секунд")
    for i, cand in enumerate(candidates, 1):
        print(f"\nКандидат {i}:")
        print(f"  Сообщения: {cand.user_messages}")
        # Если есть возможность посмотреть трассу, можно её вывести
        # (но это зависит от SDK)

if __name__ == "__main__":
    main()