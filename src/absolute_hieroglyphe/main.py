import os

col_names = ["Type_monu", "Mains", "Forme_main", "Doigts", "Doigts_indiv", "Espacement doigts", "Nombre doigts", "Symétrie", "Pouces_distingues", "Angle des coudes", "Type de jointure", "Technique", "Pavois", "Couleur", "Bracelets", "Combinaison"]
col_names = ["Pouces_distingues"]

skip_dataset = False
expansion_rate = 10
skip_training = False
seed = 92
epochs = 4
batch_size = 16
skip_testing = False
test_image = "data/Dataset_compl._MB/abydos_ramsesII_1.jpg"
test_folder = "data/Dataset_compl._MB/"

for col_name in col_names:

    print("--- Starting splitter ---")
    from absolute_hieroglyphe import splitter
    col_index = splitter.split(col_name, skip_dataset)
    print("--- Splitter finished ---")

    if not skip_dataset:
        print(f"--- Starting image expansion ---")
        from absolute_hieroglyphe import agument
        agument.transform.set_random_seed(seed)
        for folder in os.listdir("dataset"):
            directory = os.path.join("dataset/" + folder)
            agument.augment_folder(directory, directory, agument.transform, expansion_rate)
        print("--- Expansion finished ---")
    else:
        print("--- Skipping image expansion ---")

    if not skip_training:
        print(f"--- Starting training for {col_name} ({col_index}) ---")
        from absolute_hieroglyphe import trainer
        trainer.train(seed, epochs, batch_size, col_index, col_name)
        print("--- Trainning finished ---")
    else:
        print("--- Skipping training ---")

    if not skip_testing:
        print(f"--- Starting tester for {col_name} ({col_index}) ---")
        from absolute_hieroglyphe import tester
        tester.test(seed, epochs, batch_size, col_index, col_name, test_image, test_folder)
        print("--- Testing finished ---")
    else:
        print("--- Skipping testing ---")
