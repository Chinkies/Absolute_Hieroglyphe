col_index = 9 # colonne "Pouces_distingues"
seed = 92
epochs = 4
batch_size = 16

print("--- Starting spliter ---")
import splitter
splitter.split(col_index)
print("--- Spliter finished ---")

# TODO call data expension here


print("--- Starting training ---")
import trainer
trainer.train(seed, epochs, batch_size, col_index)
print("--- Trainning finished ---")
