import yaml
from mushroom_rl.environments import CarOnHill, Gym


class exp:
    def __init__(self, mdp_name:str):
        self.__get_mdp(mdp_name)

    def __get_mdp(self, mdp_name: str):
        if mdp_name == "CarOnHill":
            self.mdp = CarOnHill()
            game_config = {"gamma": 1}
        else:
            game_name = mdp_name.split("-")[0]
            with open("src/config.yaml","r",) as f:
                game_config = yaml.load(f, yaml.Loader)[game_name]
            if "horizon" not in game_config.keys():
                self.gamma = game_config["gamma"]
                self.mdp = Gym(game_name, gamma=self.gamma)
            else:
                self.gamma = game_config["gamma"]
                self.mdp = Gym(
                    game_name,
                    horizon=game_config["horizon"],
                    gamma=self.gamma,
                )
