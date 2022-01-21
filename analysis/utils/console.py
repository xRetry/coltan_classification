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


def print_cv_summary(cv_results_steps: list):
    warnings = []
    n_longest = 15
    for s, cv_results in enumerate(cv_results_steps):
        warnings.append([])
        for m, eval_result_model in enumerate(cv_results.eval_results):
            warnings[s].append({})
            for eval_result in eval_result_model:
                for w in eval_result.warnings:
                    msg = w.message.args[0]
                    n = warnings[s][m].get(msg)
                    if n is None:
                        n = 0
                    warnings[s][m][msg] = n + 1
                    if len(msg) > n_longest:
                        n_longest = len(msg)+len(str(n+1))

    n_spacer = (n_longest + 10 - 9) // 2 + 1
    spacer = ''.join(['=' for i in range(n_spacer)])
    print('\n{} Summary {}'.format(spacer, spacer))
    for s in range(len(warnings)):
        print('Step {}:'.format(s+1))
        for m in range(len(warnings[s])):
            print('   Model {}:'.format(m+1))  # 3
            for msg, n in warnings[s][m].items():
                print('      {} - x{}'.format(msg, n))
    print('{}'.format(''.join(['=' for i in range(n_spacer * 2 + 9)])))


if __name__ == '__main__':
    pass
