col_names = ["Type_monu", "Mains", "Forme_main", "Doigts", "Doigts_indiv", "Espacement doigts", "Nombre doigts", "Symétrie", "Pouces_distingues", "Angle des coudes", "Type de jointure", "Technique", "Pavois", "Couleur", "Bracelets", "Combinaison"]
col_names = ["Pouces_distingues"]

seed = 92
epochs = 4
batch_size = 16

for col_name in col_names:

    print("--- Starting spliter ---")
    import splitter
    col_index = splitter.split(col_name, False)
    print("--- Spliter finished ---")

    # TODO call data expension here

    print("--- Starting training ---")
    import trainer
    trainer.train(seed, epochs, batch_size, col_index)
    print("--- Trainning finished ---")
