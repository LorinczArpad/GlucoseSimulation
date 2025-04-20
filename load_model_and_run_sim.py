import os
from pathlib import Path
from stable_baselines3 import A2C, TD3

from simulation_core import (
    SimulationConfig, MealGenerator, EnvironmentManager,
    ModelTrainer, SimulationRunner, DataSaver, MetricsCalculator, 
    clear_console
)

def main():
    clear_console()
    print("=" * 60)
    print("     Interactive Model Loader & Simulation Runner")
    print("=" * 60)

    model_type = input("Choose model type (A2C / TD3): ").strip().upper()
    if model_type not in {"A2C", "TD3"}:
        model_type = "A2C"

    use_existing_models = input("Do you want to load existing trained models? (y/n): ").strip().lower() == "y"

    config = SimulationConfig(model_type=model_type)
    patient_params = config.get_patient_params()
    print(f"Patient {config.patient_name} | BW: {patient_params['bw']} kg")

    meal_gen = MealGenerator(config)
    scenario, meals = meal_gen.create_meal_scenario(patient_params["bw"])
    meal_gen.print_meals(meals)

    env_mgr = EnvironmentManager(config, scenario)
    env_mgr.register_environments()
    env, lowenv, innerenv, highenv = env_mgr.create_environments()

    trainer = ModelTrainer(lowenv, innerenv, highenv, config)
    lowmodel, innermodel, highmodel = trainer.train_or_load_models(use_existing_models)

    runner = SimulationRunner(env, lowmodel, innermodel, highmodel, config)
    frames, log_data = runner.run()

    saver = DataSaver(env_mgr.path_to_results, config)
    saver.save_csv(log_data)
    saver.save_video(frames)

    metrics_calc = MetricsCalculator(env_mgr.path_to_results)
    metrics = metrics_calc.calculate(log_data)
    metrics_calc.save(metrics)

    env.close()


if __name__ == "__main__":
    main()
