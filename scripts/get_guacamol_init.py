import torch 
import pandas as pd 
import selfies as sf 
from guacamol import standard_benchmarks
from torch.utils.data import TensorDataset, DataLoader

from aabo.tasks.utils.selfies_vae.data import SELFIESDataset
from aabo.tasks.utils.selfies_vae.model_positional_unbounded import InfoTransformerVAE
from aabo.tasks.utils.selfies_vae.data import collate_fn

med1 = standard_benchmarks.median_camphor_menthol() #'Median molecules 1'
med2 = standard_benchmarks.median_tadalafil_sildenafil() #'Median molecules 2',
pdop = standard_benchmarks.perindopril_rings() # 'Perindopril MPO',
osmb = standard_benchmarks.hard_osimertinib()  # 'Osimertinib MPO',
adip = standard_benchmarks.amlodipine_rings()  # 'Amlodipine MPO' 
siga = standard_benchmarks.sitagliptin_replacement() #'Sitagliptin MPO'
zale = standard_benchmarks.zaleplon_with_other_formula() # 'Zaleplon MPO'
valt = standard_benchmarks.valsartan_smarts()  #'Valsartan SMARTS',
dhop = standard_benchmarks.decoration_hop() # 'Deco Hop'
shop = standard_benchmarks.scaffold_hop() # Scaffold Hop'
rano= standard_benchmarks.ranolazine_mpo() #'Ranolazine MPO' 
fexo = standard_benchmarks.hard_fexofenadine() # 'Fexofenadine MPO'... 'make fexofenadine less greasy'

guacamol_objs = {
    "med1":med1,"pdop":pdop, 
    "adip":adip, "rano":rano, 
    "osmb":osmb, "siga":siga, 
    "zale":zale, "valt":valt,
    "med2":med2,"dhop":dhop, 
    "shop":shop, 'fexo':fexo
} 

def save_decoded_smiles(
    N=20_000,
    bsz=128,
    path_save_scores="../tasks/utils/selfies_vae/train_ys_v2.csv", 
):
    vae, dataobj = initialize_vae()
    z = torch.load("../tasks/utils/selfies_vae/train_zs.pt")
    z = z[0:N]
    decoded_smiles = zs_to_smiles(z, vae, dataobj, bsz=bsz)
    save_df = {}
    save_df["smile"] = decoded_smiles
    save_df = pd.DataFrame.from_dict(save_df)
    save_df.to_csv(path_save_scores, index=False)

def save_decoded_scores(
    path_save_scores="../tasks/utils/selfies_vae/train_ys_v2.csv",
    relevant_guac_tasks=["osmb", "med1", "med2", "fexo"], 
):
    new_df = {}
    df = pd.read_csv(path_save_scores)
    smiles = df["smile"].values.tolist()
    new_df["smile"] = smiles 
    for task_id in relevant_guac_tasks:
        guacamol_obj_func = guacamol_objs[task_id].objective
        new_df[task_id] = []
        for smile_str in smiles:
            score = guacamol_obj_func.score(smile_str)
            new_df[task_id].append(score)
        save_df = pd.DataFrame.from_dict(new_df)
        save_df.to_csv(path_save_scores, index=False)
        print("finished saving scores for task: ", task_id)



def zs_to_smiles(z, vae, dataobj, bsz=128):
    decoded_selfies = []
    train_dataset = TensorDataset(z)
    train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=False)
    check = True 
    for (z_batch, ) in train_loader:
        with torch.no_grad():
            sample = vae.sample(z=z_batch.reshape(-1, 2, 128).cuda())
        decoded_selfies_batch = [dataobj.decode(sample[i]) for i in range(sample.size(-2))]
        decoded_selfies = decoded_selfies + decoded_selfies_batch
        if check:
            print(f"BATCH DECODED SUCCESSFULLY, BSZ = {bsz} works")
            check = False
    decoded_smiles = []
    for selfie in decoded_selfies:
        smile = sf.decoder(selfie)
        decoded_smiles.append(smile)
    return decoded_smiles


def initialize_vae(path_to_vae_statedict="../tasks/utils/selfies_vae/selfies-vae-state-dict.pt"):
    ''' Sets self.vae to the desired pretrained vae and 
        sets self.dataobj to the corresponding data class 
        used to tokenize inputs, etc. '''
    dataobj = SELFIESDataset()
    vae = InfoTransformerVAE(dataset=dataobj)
    state_dict = torch.load(path_to_vae_statedict) 
    vae.load_state_dict(state_dict, strict=True) 
    vae = vae.cuda()
    vae = vae.eval()
    return vae, dataobj 


def get_tokenized_selfies(selfies_list, dataobj):
    X_list = []
    for selfie_str in selfies_list:
        tokenized_selfie = dataobj.tokenize_selfies([selfie_str])[0]
        encoded_selfie = dataobj.encode(tokenized_selfie).unsqueeze(0)
        X_list.append(encoded_selfie)
    X = collate_fn(X_list)
    return X



if __name__ == "__main__":
    save_decoded_smiles(
        N=10_000,
        bsz=128,
        path_save_scores="../tasks/utils/selfies_vae/train_ys_v2.csv", 
    )
    print("All decoded smiles saved, now saving scores...")
    save_decoded_scores(
        path_save_scores="../tasks/utils/selfies_vae/train_ys_v2.csv",
        relevant_guac_tasks=["osmb", "med1", "med2", "fexo", "adip", "siga", "zale", "pdop", "rano"], 
    )
    print("ALL SCORES SAVED, DONE")

