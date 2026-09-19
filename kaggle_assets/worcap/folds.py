"""LOYO com embargo +-1 ano sobre 1940-2022, holdout 2018-2022 congelado."""
from .config import TRAIN_START, TRAIN_END, HOLDOUT_YEARS


def loyo_folds(embargo: int = 1, holdout=HOLDOUT_YEARS):
    """Yield dicts {fold, test_year, train_years}.

    - test_year percorre TRAIN_START..TRAIN_END excluindo holdout.
    - train_years exclui test_year +- embargo e todos os anos de holdout.
    """
    holdout = set(holdout or [])
    folds = []
    for y in range(TRAIN_START, TRAIN_END + 1):
        if y in holdout:
            continue
        ban = {y - embargo, y, y + embargo} | holdout
        train = [t for t in range(TRAIN_START, TRAIN_END + 1) if t not in ban]
        folds.append({"fold": y, "test_year": y, "train_years": train})
    return folds


def final_fit_years(holdout=HOLDOUT_YEARS):
    """Anos usados no fit final: tudo exceto holdout."""
    holdout = set(holdout or [])
    return [y for y in range(TRAIN_START, TRAIN_END + 1) if y not in holdout]
