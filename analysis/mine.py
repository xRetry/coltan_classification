import numpy as np
from classes.dataset import Dataset, Sample
from classes.parameters import Parameters
from classes.mines import Mine
from functions import plotting
from typing import List


class MineAnalyser:
    @staticmethod
    def parameter_progression(parameters: Parameters) -> None:
        """
        Progressively adds samples to mine and collects the parameters. Plots the result.
        """
        # Load dataset
        dataset = Dataset()
        samples_mine_selected, _ = MineAnalyser._select_mine_samples(dataset)
        # Build mine
        mine = MineAnalyser._create_mine(parameters)
        # Adding samples to mine and collecting parameters
        mine_params = []
        for sample in samples_mine_selected:
            mine.add_sample(sample)
            mine_params.append(mine.parameters)
        # Plotting the mine parameters over time
        plotting.plot_progression(mine_params)

    @staticmethod
    def evaluation(parameters: Parameters):
        """
        Evaluates all samples using a mine and plots the result.
        """
        # Load dataset
        dataset = Dataset()
        # Create mine
        mine = MineAnalyser._create_mine(parameters)
        # Get samples to add to mine
        samples_mine_selected, mine_id_selected = MineAnalyser._select_mine_samples(dataset)
        # Add selected samples to mine
        for sample in samples_mine_selected:
            mine.add_sample(sample)
        # Evaluate all samples in dataset and collect the results
        eval_values, labels, has_sample = np.zeros(len(dataset)), np.zeros(len(dataset)), np.zeros(len(dataset), dtype=bool)
        for i, sample in enumerate(dataset):
            if sample.mine_id == mine_id_selected:
                has_sample[i] = True
            eval_values[i] = mine.eval_sample(sample)
            labels[i] = sample.proportional_score
        # Plot the result
        plotting.plot_mine_evaluation(eval_values, labels, has_sample)

    @staticmethod
    def _select_mine_samples(dataset: Dataset) -> (List[Sample], str):
        """
        Returns all samples from the mine with most samples.
        """
        # Get unique mine IDs and the amount of samples per mine
        mine_ids_unique, mine_ids_count = np.unique([smp.mine_id for smp in dataset], return_counts=True)
        # Select mine with most samples for analysis
        idx_selected = np.argmax(mine_ids_count)
        samples_mine_selected = [sample for sample in dataset if sample.mine_id == mine_ids_unique[idx_selected]]
        return samples_mine_selected, mine_ids_unique[idx_selected]

    @staticmethod
    def _create_mine(parameters: Parameters) -> Mine:
        """
        Builds mine according to provided mine parameters.
        """
        mine = parameters.MineClass(
            coordinates=np.zeros(3),
            status=0,
            parameters=parameters,
            **parameters.mine_kwargs
        )
        return mine


if __name__ == '__main__':
    pass
