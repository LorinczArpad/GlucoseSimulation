from src.ModelTraining.ModelTrainer import ModelTrainer
from src.ModelTraining.SimulationRunner import SimulationRunner
from src.SimulationPreparation.MealGenerator import MealGenerator
from src.SimulationPreparation.SimulationConfig import SimulationConfig
from src.SimulationPreparation.EnviromentManager import EnvironmentManager


def main():
    # Setup Config 
    config = SimulationConfig(model_type="PPO")
    patient_params = config.get_patient_params()
    print(f"Body weight for {config.patient_name}: {patient_params['bw']} kg")
    #Generate Meals
    meal_generator = MealGenerator(config)
    meal_scenario, meals = meal_generator.create_meal_scenario(patient_params["bw"])
    meal_generator.print_meals(meals)
    # Manage Enviroments
    env_manager = EnvironmentManager(config, meal_scenario)
    env_manager.register_environments()
    env, lowenv, innerenv, highenv = env_manager.create_environments()
    # Train Models 
    trainer = ModelTrainer(lowenv, innerenv, highenv, config)
    best_params = trainer.tune_models()
    lowmodel, innermodel, highmodel = trainer.train_models(best_params)
    # Run the simulation
    runner = SimulationRunner(env, lowmodel, innermodel, highmodel, config)
    frames, log_data, truncated = runner.run()
    # Save Result and metrics
    saver = ModelTrainer.DataSaver(env_manager.path_to_results, config)
    metrics_calculator = ModelTrainer.MetricsCalculator(env_manager.path_to_results, config)
    metrics = metrics_calculator.calculate_metrics(log_data)
    metrics_calculator.save_metrics(metrics)
    saver.save_video(frames)
    saver.save_csv(log_data)
    env.close() 

if __name__ == "__main__":
    main()