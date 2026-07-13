"""
Gemeinsame Split-Logik für Train/Val/Test.

WICHTIG: train.py und sample.py müssen exakt denselben random_state,
val_size und test_size verwenden -- sonst driften die Splits auseinander
und sample.py wertet plötzlich (teilweise) auf Trainingsdaten aus.
Deshalb werden diese drei Werte in train.py als CLI-Args geführt und
landen automatisch in config.json; sample.py liest sie von dort statt
sie neu zu raten.
"""

from sklearn.model_selection import train_test_split


def make_splits(X_full, Y_full, random_state=42, val_size=0.2, test_size=0.2):
    """
    Zwei nacheinander ausgeführte train_test_split-Aufrufe, beide mit
    demselben random_state:

      1) Test wird zuerst vom kompletten Set abgespalten (Anteil test_size).
      2) Val wird vom verbleibenden Rest abgespalten (Anteil val_size von diesem Rest).

    Bei den Defaults (val_size=0.2, test_size=0.2) ergibt das ungefähr
    64% Train / 16% Val / 20% Test.

    returns: (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
    """
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X_full, Y_full, test_size=test_size, random_state=random_state
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval, test_size=val_size, random_state=random_state
    )
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)