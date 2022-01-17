from typing import Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class ProgressBar:
    bars: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # {name: (idx_current, n_iter)}
    post_fix: str = ''

    def add_bar(self, name, idx_current, n_iter):
        self.bars[name] = (idx_current, n_iter)

    def print(self, postfix: str='', is_end: bool=True) -> None:
        """
        Prints progress bars from items of dictionary. Format: {name: (current progress, full length)}.
        """
        # Iterating through progress bars and building console output
        output = ''
        for k, v in self.bars.items():
            # Determine amount of current and completed elements
            n_load = int((v[0] + 1) / v[1] * 10)
            n_full = int((v[0]) / v[1] * 10) if not is_end else n_load
            # Creating line string
            line = '=' * n_full + '-' * (n_load - n_full) + ' ' * (10 - n_load)
            # Adding line string to console output
            output += '{}: |{}| {}/{}\t\t'.format(k, line, v[0] + 1 if is_end else v[0], v[1])
        # Printing output to console
        print('\r{}\t\t{}'.format(output, postfix), end='')


if __name__ == '__main__':
    pass
