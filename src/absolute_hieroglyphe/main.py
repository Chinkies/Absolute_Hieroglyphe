import os
from absolute_hieroglyphe import agument
from absolute_hieroglyphe import splitter
from absolute_hieroglyphe import trainer

col_names = ["Type_monu", "Mains", "Forme_main", "Doigts", "Doigts_indiv", "Espacement doigts", "Nombre doigts", "Symétrie", "Pouces_distingues", "Angle des coudes", "Type de jointure", "Technique", "Pavois", "Couleur", "Bracelets", "Combinaison"]
col_names = ["Pouces_distingues"]

rebuild_dataset = True
expansion_rate = 10
seed = 92
epochs = 4
batch_size = 16

for col_name in col_names:

    print("--- Starting splitter ---")
    col_index = splitter.split(col_name, not rebuild_dataset)
    print("--- Splitter finished ---")

    if rebuild_dataset:
        print(f"--- Starting image expansion ---")
        agument.transform.set_random_seed(seed)
        for folder in os.listdir("dataset"):
            directory = os.path.join("dataset/" + folder)
            agument.augment_folder(directory, directory, agument.transform, expansion_rate)
        print("--- Expansion finished ---")
    else:
        print("--- Skipping image expansion ---")

    print(f"--- Starting training for {col_name} ({col_index}) ---")
    trainer.train(seed, epochs, batch_size, col_index, col_name)
    print("--- Trainning finished ---")
