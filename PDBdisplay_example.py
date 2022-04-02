##EXAMPLE1:
#------------------------------------------------------------------
# 1. Initialize a Molecule object ---------------------------------
fn_pro = './example_files/1f5l_protein.pdb' #path to protein PDB file
fn_lig = './example_files/1f5l_ligand.pdb' #path to ligand PDB file
mm = Molecule(fn_pro_PDB = fn_pro, fn_lig_PDB = fn_lig, ID = ID, sant = True)

# 2. Display the protein ------------------------------------------
# 2.1. Display atoms in the protein -------------------------------
mm.Plot_atoms(heavyAtm = True, tp = 'Symbol', tp_sup = None)
mm.Plot_atoms(heavyAtm = True, tp = 'Symbol', tp_sup = 'C')
mm.Plot_atoms(clr = 'crimson', heavyAtm = True, tp = 'Ring', tp_sup = 1)
mm.Plot_atoms(clr = 'gold', heavyAtm = True, tp = 'FC', tp_sup = None)

# 2.2. Display residues in the protein ----------------------------
mm.Plot_residues_in_chain(clr = None, heavyAtm = True, res_filter = None, bkbn = True)
mm.Plot_residues_in_chain(clr = 'gold', heavyAtm = True, res_filter = None, bkbn = True)
mm.Plot_residues_in_chain(clr = 'pink', heavyAtm = True, res_filter = None, bkbn = False)
mm.Plot_residues_in_chain(clr = 'green', heavyAtm = True, res_filter = 'ALA', bkbn = False)

#2.3. Display surface of the protein ------------------------------
mm.Plot_surf(clr = 'crimson', heavyAtm = True, alpha = 2)
mm.Plot_surf(clr = 'crimson', heavyAtm = True, alpha = 3)
mm.Plot_surf(clr = 'crimson', heavyAtm = True, alpha = 10)

#3. Display protein-ligand interface ------------------------------
mm.Plot_interface(clr = 'gold', heavyAtm = True, alpha = None, cutoff = 4.5)
mm.Plot_interface(clr = 'gold', heavyAtm = True, alpha = 3, cutoff = 4.5)

# save figure -----------------------------------------------------
plt.savefig('display.png')   # save the current figure to file
plt.close() 






    
