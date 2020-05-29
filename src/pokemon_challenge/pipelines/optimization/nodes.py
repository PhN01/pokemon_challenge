import logging
from typing import Any, Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import ortools
from ortools.linear_solver import pywraplp

from tqdm import tqdm
import time
import random


def lp_optimization(available_performance: pd.DataFrame, submission_file: pd.DataFrame):
    log = logging.getLogger(__name__)

    # prepare array with optimization metric (HPPR)
    battle_outcomes = available_performance[
        [
            "hppr_caterpie",
            "hppr_golem",
            "hppr_krabby",
            "hppr_mewtwo",
            "hppr_raichu",
            "hppr_venusaur",
        ]
    ]
    battle_outcomes = np.array(battle_outcomes)

    num_enemy_pokemon = 6
    num_available_pokemon = len(battle_outcomes)
    enemies = ["Caterpie", "Golem", "Krabby", "Mewtwo", "Raichu", "Venusaur"]

    prices = available_performance["Price_1"]
    prices = np.array(prices)

    max_budget = 3500

    # Create the mip solver with the CBC backend.
    solver = pywraplp.Solver(
        "pokemon_mip_problem", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
    )

    # definition of variables
    x = {}

    for i in range(num_available_pokemon):
        for j in range(num_enemy_pokemon):
            x[i, j] = solver.BoolVar("x[%i,%i]" % (i, j))

    # objective
    solver.Maximize(
        solver.Sum(
            [
                battle_outcomes[i][j] * x[i, j]
                for i in range(num_available_pokemon)
                for j in range(num_enemy_pokemon)
            ]
        )
    )

    # constraints

    # Each available pokemon is assigned to at most 1 enemy pokemon.
    for n in range(int(num_available_pokemon / 10)):
        solver.Add(
            solver.Sum(
                [
                    x[i, j]
                    for j in range(num_enemy_pokemon)
                    for i in range(n * 10, (n + 1) * 10)
                ]
            )
            <= 1
        )

    # Each enemy pokemon gets assigned to exactly one available pokemon.
    for j in range(num_enemy_pokemon):
        solver.Add(solver.Sum([x[i, j] for i in range(num_available_pokemon)]) == 1)

    # The price of the selected pokemon must be at most equal to the budget
    solver.Add(
        solver.Sum(
            [
                x[i, j] * prices[i]
                for i in range(num_available_pokemon)
                for j in range(num_enemy_pokemon)
            ]
        )
        <= max_budget
    )

    sol = solver.Solve()

    final_objective = solver.Objective().Value()

    log.info(f"Optimization Finished.  Time = {solver.WallTime()} milliseconds")

    final_objective_norm = final_objective / num_enemy_pokemon

    log.info(f"Average HPPR = {final_objective_norm}")
    for j in range(num_enemy_pokemon):
        for i in range(num_available_pokemon):
            if x[i, j].solution_value() > 0:
                submission_file.loc[
                    j, "SelectedPokemonID"
                ] = available_performance.SelectedPokemonID[i]
                log.info(
                    f"Pokemon {available_performance.Name_1[i]} (Level {available_performance.Level_1[i]}) assigned "
                    f"to enemy {enemies[j]}.  HPPR = {battle_outcomes[i,j]}  Price = {prices[i]}"
                )

    return submission_file


class GeneticPokemons(object):
    def __init__(
        self, data, num_lineups, duration=60, opt_metric="result", p_mutation=0.2
    ):
        self.data = data
        self.num_lineups = num_lineups
        self.duration = duration
        self.n_pokemons = None
        self.top_150 = []
        self.all_pokemons = []
        self.opt_metric = opt_metric
        self.p_mutation = p_mutation

    def run(self):

        runtime = time.time() + self.duration
        while time.time() < runtime:
            self.get_lineups()
            self.top_150.sort(key=lambda x: x[-1], reverse=True)
            # We use 150 as the number here b/c drafkings only allows a maximum of 150 lineups in any one contest
            self.top_150 = self.top_150[:150]

    def get_lineups(self):

        # generate 10 new lineups
        new_lineups = [self.generate_lineup() for _ in range(10)]

        # sort the lineups by their predicted score
        new_lineups.sort(key=lambda x: x[-1], reverse=True)

        # Add the newly created lineups to the top_150 (they will be sorted and bottom ones removed later)
        self.top_150.extend(new_lineups)

        # Mate the top 3 lineups together
        offspring_1 = self.mate_lineups(new_lineups[0], new_lineups[1])
        offspring_2 = self.mate_lineups(new_lineups[0], new_lineups[2])
        offspring_3 = self.mate_lineups(new_lineups[1], new_lineups[2])

        # Mate the offspring with a randomly selected lineup from the top_150 and add to top_150
        # Adding this step makes the algorithm more greedy, and produces higher projections, but can be skipped
        self.top_150.append(
            self.mate_lineups(
                offspring_1, self.top_150[random.randint(0, len(self.top_150) - 1)]
            )
        )
        self.top_150.append(
            self.mate_lineups(
                offspring_2, self.top_150[random.randint(0, len(self.top_150) - 1)]
            )
        )
        self.top_150.append(
            self.mate_lineups(
                offspring_3, self.top_150[random.randint(0, len(self.top_150) - 1)]
            )
        )

        # Add the original offspring to the top_150
        self.top_150.append(offspring_1)
        self.top_150.append(offspring_2)
        self.top_150.append(offspring_3)

    def mate_lineups(self, lineup1, lineup2):
        """
            Mate takes in two lineups, creates lists of positions from the available lineups
            and adds another random player to each position list. It then randomly selects the required
            number of positions for a new lineup from each position list, checks that the lineup is valid,
            and returns the valid lineup.
        """

        # Create lists of all available players for each position from the two lineups plus a random player or 2

        # caterpie = [lineup1[0], lineup2[0], self.all_pokemons[random.randint(0, self.n_pokemons - 1)]]
        # golem = [lineup1[1], lineup2[1], self.all_pokemons[random.randint(0, self.n_pokemons - 1)]]
        # krabby = [lineup1[2], lineup2[2], self.all_pokemons[random.randint(0, self.n_pokemons - 1)]]
        # mewtwo = [lineup1[3], lineup2[3], self.all_pokemons[random.randint(0, self.n_pokemons - 1)]]
        # raichu = [lineup1[4], lineup2[4], self.all_pokemons[random.randint(0, self.n_pokemons - 1)]]
        # venusaur = [lineup1[5], lineup2[5], self.all_pokemons[random.randint(0, self.n_pokemons - 1)]]
        #
        #
        # # Randomly grab num players from each position to fill out the new mated lineup
        # def grab_pokemon(pokemons):
        #     avail_pokemons = copy.deepcopy(pokemons)
        #     i = random.randint(0, len(avail_pokemons) - 1)
        #     return avail_pokemons[i]
        #
        while True:
            # Create the new lineup by selecting players from each position list
            cutoff = int(np.random.choice(np.arange(6), size=1))
            lineup = lineup1[:cutoff] + lineup2[cutoff:6]

            def mutation(pokemon):
                mutate = np.random.choice(
                    [0, 1], p=[1 - self.p_mutation, self.p_mutation]
                )
                if not mutate:
                    return pokemon
                else:
                    return self.all_pokemons[random.randint(0, self.n_pokemons - 1)]

            lineup = [mutation(pokemon) for pokemon in lineup]

            # Check if the lineup is valid (i.e. it satisfies some basic constraints)
            lineup = self.check_valid(lineup)
            # If lineup isn't valid, run mate again
            if lineup:
                return lineup

    def generate_lineup(self):
        """
            Generate a single lineup with the correct amount of positional requirements.
            The lineup is then checked for validity and returned.
        """

        while True:
            # add the correct number of each position to a lineup
            lineup = []
            for _ in range(6):
                lineup.append(self.all_pokemons[random.randint(0, self.n_pokemons - 1)])

            # Check if the lineup is valid (i.e. it satisfies some basic constraints)
            lineup = self.check_valid(lineup)
            if lineup:
                return lineup

    def check_valid(self, lineup):

        # calculate the total projection of the lineup based on player averages
        result_list = []
        hppr_list = []
        cost_list = []
        names_list = []
        for i, opponent in enumerate(
            ["caterpie", "golem", "krabby", "mewtwo", "raichu", "venusaur"]
        ):
            result_list.append(lineup[i][opponent])
            hppr_list.append(np.maximum(lineup[i][opponent], 0) / lineup[i]["hp"])
            cost_list.append(lineup[i]["price"])
            names_list.append(lineup[i]["name"])
        result = np.mean(result_list)
        hppr = np.mean(hppr_list)
        cost = np.sum(cost_list)
        indiv = len(set(names_list))

        if (cost < 3500) and (indiv == 6):
            # add the salary and the projection to the lineup of players and return the lineup
            if self.opt_metric == "result":
                lineup.extend((cost, hppr, result))
            else:
                lineup.extend((cost, result, hppr))
            return lineup
        return False

    def prepare_data(self):
        """
            Load in the roster of players from the DraftKings DKSalaries CSV download.
            Removes players with an average of 0 points, indicating they are probably not active.
            The user should do additional filters to remove players who are not starting, injured, or
            not active. The user should also replace the average points with their own projections.
        """

        pokemon_df = self.data

        for i, row in pokemon_df.iterrows():
            pokemon = {}
            pokemon["name"] = row["Name_1"]
            pokemon["level"] = row["Level_1"]
            pokemon["price"] = row["Price_1"]
            pokemon["hp"] = row["HP_1"]
            pokemon["id"] = row["SelectedPokemonID"]
            pokemon["caterpie"] = row["res_caterpie"]
            pokemon["golem"] = row["res_golem"]
            pokemon["krabby"] = row["res_krabby"]
            pokemon["mewtwo"] = row["res_mewtwo"]
            pokemon["raichu"] = row["res_raichu"]
            pokemon["venusaur"] = row["res_venusaur"]
            self.all_pokemons.append(pokemon)

        self.n_pokemons = len(self.all_pokemons)


def ga_optimization(available_performance: pd.DataFrame, submission_file: pd.DataFrame):
    log = logging.getLogger(__name__)

    enemies = ["caterpie", "golem", "krabby", "mewtwo", "raichu", "venusaur"]

    ga_runtime = 6000  # 1h
    ga_lineups = 150

    ga = GeneticPokemons(
        data=available_performance,
        num_lineups=ga_lineups,
        duration=ga_runtime,
        opt_metric="hppr",
    )
    ga.prepare_data()
    ga.run()

    best_lineup = ga.top_150[0]
    final_objective = best_lineup[-1]
    total_cost = best_lineup[-3]
    for i, pokemon in enumerate(best_lineup[:6]):
        submission_file.loc[i, "SelectedPokemonID"] = pokemon["id"]
        log.info(
            f"Pokemon {pokemon['name']} (Level {pokemon['level']}) assigned "
            f"to enemy {enemies[i]}. Result = {pokemon[f'{enemies[i]}']}  Price = {pokemon['price']}"
        )

    log.info(f"Average HPPR = {final_objective}")
    log.info(f"Total Cost = {total_cost}")

    log.info(f"Time = {ga_runtime} seconds")

    return submission_file
