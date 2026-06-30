from src.utils.episode_generation_protocol import LatentSpace


input_params = {}
input_params["day_length"] = 80
input_params["mean_duration"] = 5
input_params["fixed_duration"] = True
input_params["num_swaps"] = 4

latent_specs = {}
latent_specs["num"] = 2
latent_specs["total_sizes"] = [50, 50]
latent_specs["act_sizes"] = [10, 10]
latent_specs["dims"] = [5, 5]
latent_specs["prob_list"] = [0.5 / 5 if i == j else 0.5 / 20 for i in range(5) for j in range(5)]


num_days = {}
num_days["warm_up"] = 100
num_days["test"] = 100
num_days["full_learning_semantic_load_1"] = 500
num_days["full_learning_semantic_load_2"] = 500
