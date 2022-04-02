#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 21:53:50 2021

@author: debbywang
"""
from rdkit import Chem
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from itertools import chain
import time

# 1. read a PDB file into a tuple (xyz,mol) ---------------------------------------
def load_molecule(molecule_file, sanitize=True):
  """Converts molecule file to (xyz-coords, obmol object)
  Given molecule_file, returns a tuple of xyz coords of molecule
  and an rdkit object representing that molecule in that order `(xyz,
  rdkit_mol)`. This ordering convention is used in the code in a few
  places.
  Parameters
  ----------
  molecule_file: str
    filename for molecule
  sanitize: bool, optional (default False)
    If True, sanitize molecules via rdkit
  Returns
  -------
  Tuple (xyz, mol) if file contains single molecule. Else returns a
  list of the tuples for the separate molecules in this list.
  Note
  ----
  This function requires RDKit to be installed.
  """
  if ".mol2" in molecule_file:
    my_mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
  elif ".sdf" in molecule_file:
    suppl = Chem.SDMolSupplier(str(molecule_file), sanitize=False)
    my_mol = suppl[0]
  elif ".pdb" in molecule_file:
    my_mol = Chem.MolFromPDBFile(
        str(molecule_file), sanitize=False, removeHs=False)
  else:
    raise ValueError("Unrecognized file type for %s" % str(molecule_file))

  if my_mol is None:
    raise ValueError("Unable to read non None Molecule Object")

  if sanitize:
    try:
      Chem.SanitizeMol(my_mol)
    except:
      print("Mol %s failed sanitization" % Chem.MolToSmiles(my_mol))

  xyz = np.zeros((my_mol.GetNumAtoms(), 3))
  conf = my_mol.GetConformer()
  for i in range(conf.GetNumAtoms()):
    position = conf.GetAtomPosition(i)
    xyz[i, 0] = position.x
    xyz[i, 1] = position.y
    xyz[i, 2] = position.z

  return xyz, my_mol

##EXAMPLE:
## 读入PDB文件并计算距离矩阵 ---------------------------------------
#ID = '1f5l'
#fn_pro = 'C://Users/debby/Desktop/' + ID + '/' + ID + '_protein.pdb' #蛋白pdb文件的路径
#fn_lig = 'C://Users/debby/Desktop/' + ID + '/' + ID + '_ligand.pdb' #配体pdb文件的路径
#pro = load_molecule(molecule_file = fn_pro, sanitize=True)
## pro[0] is the 3D-coordinates of the molecule, pro[1] is the molecule with atom properties
#lig = load_molecule(molecule_file = fn_lig, sanitize=True)
#distance_matrix = cdist(pro[0], lig[0], metric = 'euclidean')
#int_cutoff = 12
#contacts = np.nonzero(distance_matrix < int_cutoff)
## -----------------------------------------------------------------

def alpha_shape_3D(pos, alpha):
    """
    Compute the alpha shape of a set of 3D points.
    Parameters:
        pos - np.array of shape (n,3) points
        alpha - alpha value
    return
        outer surface vertex indices, edge indices, and triangle indices
    """
    tetra = Delaunay(pos)
    tetrapos = np.take(pos, tetra.vertices, axis = 0)
    normsq = np.sum(tetrapos ** 2, axis = 2)[:, :, None]
    ones = np.ones((tetrapos.shape[0], tetrapos.shape[1], 1))
    a = np.linalg.det(np.concatenate((tetrapos, ones), axis = 2))
    Dx = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [1, 2]], ones), axis = 2))
    Dy = -np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 2]], ones), axis = 2))
    Dz = np.linalg.det(np.concatenate((normsq,tetrapos[:, :, [0, 1]], ones), axis = 2))
    c = np.linalg.det(np.concatenate((normsq, tetrapos), axis = 2))
    r = np.sqrt(Dx ** 2 + Dy ** 2 + Dz ** 2 - 4 * a * c) / (2 * np.abs(a))

    # Find tetrahedrals
    tetras = tetra.vertices[r < alpha,:]
    # triangles
    TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    Triangles = tetras[:, TriComb].reshape(-1, 3)
    Triangles = np.sort(Triangles, axis = 1)
    # Remove triangles that occurs twice, because they are within shapes
    TrianglesDict = defaultdict(int)
    for tri in Triangles:TrianglesDict[tuple(tri)] += 1
    Triangles=np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])
    #edges
    EdgeComb=np.array([(0, 1), (0, 2), (1, 2)])
    Edges=Triangles[:,EdgeComb].reshape(-1,2)
    Edges=np.sort(Edges,axis=1)
    Edges=np.unique(Edges,axis=0)

    Vertices = np.unique(Edges)
    return Vertices, Edges, Triangles

def alpha_shape_plot_prepare(all_pos, alpha = 3):
    """
    Preparation for alpha shape plots.
    Parameters:
        all_pos - np.array of shape (n,3) points
        alpha - alpha value
    return
        x, y, z and tri_idx for plots
    """
    vers, edges, tris = alpha_shape_3D(pos = all_pos, alpha = alpha)
    
    tricoor = np.take(all_pos, tris, axis = 0)
    tri_points = list(chain.from_iterable(tricoor))
    x, y, z = zip(*tri_points)
    tri_idx = [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(tris))]        
    
    return x, y, z, tri_idx


class Molecule(object):
    def __init__(self, fn_pro_PDB, fn_lig_PDB = None, ID = None, sant = True):
        """
        Initialize an SIFt class.
        Parameters:
            fn_pro_PDB - PDB file name of the protein
            fn_lig_PDB - PDB file name of the ligand
            ID - ID of the complex
            sant - whether to sanitize the molecule when reading in the structure file
            int_cutoff - distance threshold for identifying protein-ligand interacting atoms 
        """
        print('Reading the macromolecular structure...')
        self.ID = ID if ID is not None else "Protein"
        # read in pdb coordinates and topology
        self.pro = (load_molecule(fn_pro_PDB, sanitize=sant))
        if fn_lig_PDB is not None:
            self.lig = (load_molecule(fn_lig_PDB, sanitize=sant)) 
            # identify interacting area
            self.pd = cdist(self.pro[0], self.lig[0], metric = 'euclidean')
        else:
            self.lig = (None, None)
            self.pd = None
        # identify physico-chemical properties of atoms
        atmdict = {'Symbol': [], 'AtomicNum': [], 'Mass':[], 'FormalCharge': [], 'TotalValence': [],
                   'ConnectedHs': [], 'ConnectedOthNeighbors': [], 'IsInRing': [], 'IsAromatic': [],
                   'PDBchinID': [], 'PDBresNum': [], 'PDBresName': [], 'SecondaryStructure': [],
                   'xyz': []}
        for i in range(self.pro[0].shape[0]):            
            atom = self.pro[1].GetAtomWithIdx(int(i))
            atmdict['Symbol'] += [atom.GetSymbol()] #原子符号
            atmdict['AtomicNum'] += [atom.GetAtomicNum()] #原子序号
            atmdict['IsInRing'] += [int(atom.IsInRing())]
            atmdict['IsAromatic'] += [int(atom.GetIsAromatic())]
            atmdict['Mass'] += [atom.GetMass()] #原子质量
            atmdict['FormalCharge'] += [atom.GetFormalCharge()] #原子形式电荷
            atmdict['TotalValence'] += [atom.GetTotalValence()] #原子总的化合价
            atmdict['ConnectedHs'] += [atom.GetTotalNumHs()] #原子连接的氢原子个数            
            atmdict['ConnectedOthNeighbors'] += [atom.GetTotalDegree() - atom.GetTotalNumHs()] #原子连接的非氢原子个数
            atmdict['PDBchinID'] += [atom.GetPDBResidueInfo().GetChainId()]
            atmdict['PDBresNum'] += [atom.GetPDBResidueInfo().GetResidueNumber()]
            atmdict['PDBresName'] += [atom.GetPDBResidueInfo().GetResidueName()]
            atmdict['SecondaryStructure'] += [atom.GetPDBResidueInfo().GetSecondaryStructure()] 
            atmdict['xyz'] += [self.pro[0][i].tolist()]
        self.ProAtmProp = pd.DataFrame(atmdict)  
            
    def Plot_atoms(self, clr = 'blue', heavyAtm = True, tp = 'Symbol', tp_sup = None):
        """
        Plot specific atoms in a protein in a predefined way
        Parameters:
            clr - Color for plots
            heavyAtm - Wether to use heave atoms only (= True) or not (= False)
            tp - Atom properties to display 
                ('Symbol', 'Ring', 'Aromatic', 'Mass', 'FC', 'TotalValence', 'ConnectedHs', 'NonHneighbors')
            tp_sup - Criterion for further filter the atoms (if tp is 'Ring' or 'Aromatic', tp_sup is 1 or 0)
        """
        # filter atoms
        df = self.ProAtmProp if not heavyAtm else self.ProAtmProp.loc[(self.ProAtmProp['AtomicNum'] > 1) & (self.ProAtmProp['PDBresName'] != 'HOH')]
#        df = ProAtmProp if not heavyAtm else ProAtmProp.loc[(ProAtmProp['AtomicNum'] > 1) & (ProAtmProp['PDBresName'] != 'HOH')s]
        
        propdict = {'Symbol': ['Symbol', 'color'], 
                    'Ring': ['IsInRing', None], 
                    'Aromatic': ['IsAromatic', None], 
                    'Mass': ['Mass', 'size'], 
                    'FC': ['FormalCharge', 'size'], 
                    'TotalValence': ['TotalValence', 'size'], 
                    'ConnectedHs': ['ConnectedHs', 'size'], 
                    'NonHneighbors': ['ConnectedOthNeighbors', 'size']}
        tmp = propdict[tp]
        if tp_sup is not None:
            df = df.loc[df[tmp[0]] == tp_sup]
            
        fig = plt.figure(figsize = (12, 12))
        ax = plt.axes(projection = '3d')      
        if tp == 'Symbol':
            tmpdict = {'C': 'carbon', 'N': 'nitrogen', 'O': 'oxygen', 'S': 'sulfur', 'H': 'hydrogen'}
            for atmtp in set(df['Symbol']):
                if len(set(df['Symbol'])) > 1:
                    tit_sup = 'different'
                else:
                    tit_sup = tmpdict[atmtp]
                df_cur = df.loc[df['Symbol'] == atmtp]          
                xyz = [[item[0] for item in df_cur['xyz'].tolist()],
                       [item[1] for item in df_cur['xyz'].tolist()],
                       [item[2] for item in df_cur['xyz'].tolist()]]
                ax.scatter3D(xs = xyz[0], ys = xyz[1], zs = xyz[2], label = atmtp)
            ax.set_title('Display of %s atoms (PDB ID: 1F5L)' % tit_sup)
            ax.legend(loc = 'best')
        elif tp in ['Ring', 'Aromatic']:
            df_cur = df          
            xyz = [[item[0] for item in df_cur['xyz'].tolist()],
                   [item[1] for item in df_cur['xyz'].tolist()],
                   [item[2] for item in df_cur['xyz'].tolist()]]
            ax.scatter3D(xs = xyz[0], ys = xyz[1], zs = xyz[2], marker = 'x', c = clr)
            ax.set_title('Display of %s atoms (PDB ID: 1F5L)' % tp)
        else:
            tmpdict = {'Mass': 'mass', 'FC': 'formal charge', 'TotalValence': 'total valence', 
                       'ConnectedHs': 'hydrogen-neighbor counts', 'NonHneighbors': 'non-hydrogen-neighbors counts'}
            df_cur = df          
            xyz = [[item[0] for item in df_cur['xyz'].tolist()],
                   [item[1] for item in df_cur['xyz'].tolist()],
                   [item[2] for item in df_cur['xyz'].tolist()]]
            ax.scatter3D(xs = xyz[0], ys = xyz[1], zs = xyz[2], s = df_cur[tmp[0]] * 50, c = 'orange')
            ax.set_title('Display of %s of atoms (PDB ID: 1F5L)' % tmpdict[tp])
        plt.show()

    def Plot_residues_in_chain(self, clr = 'gold', heavyAtm = True, res_filter = 'ILE', bkbn = True):
        """
        Plot protein chains (units of residues)
        Parameters:
            clr - Color for plots
            heavyAtm - Wether to use heave atoms only (= True) or not (= False)
            res_filter - Plot a specific type of residues
            bkbn - Whether to show the backbone line
        """
        df = self.ProAtmProp if not heavyAtm else self.ProAtmProp.loc[(self.ProAtmProp['AtomicNum'] > 1) & (self.ProAtmProp['PDBresName'] != 'HOH')]
#        df = ProAtmProp if not heavyAtm else ProAtmProp.loc[(ProAtmProp['AtomicNum'] > 1) & (ProAtmProp['PDBresName'] != 'HOH')]
        
        df['x'] = [i[0] for i in df['xyz']]
        df['y'] = [i[1] for i in df['xyz']]
        df['z'] = [i[2] for i in df['xyz']]
        AAdict = {'ALA': 1, 'ARG': 2, 'ASN': 3, 'ASP': 4, 'CYS': 5, 'GLN': 6, 
                  'GLU': 7, 'GLY': 8, 'HIS': 9, 'ILE': 10, 'LEU': 11, 'LYS': 12,
                  'MET': 13, 'PHE': 14, 'PRO': 15, 'SER': 16, 'THR': 17, 'TRP': 18,
                  'TYR': 19, 'VAL': 20}
        AAdict2 = {v:k for (k, v) in AAdict.items()}
        if res_filter is not None and res_filter not in AAdict: 
            print('Wrong amino acid type!')
            return
        else:
            df['restp'] = [AAdict[i] for i in df['PDBresName']]
            chains = set(df['PDBchinID'])
            fig = plt.figure(figsize = (12, 12))
            ax = plt.axes(projection = '3d')      
            for cid in chains:
                if res_filter is not None:
                    df = df.loc[df['PDBresName'] == res_filter]
                df_tmp = df.loc[df['PDBchinID'] == cid, ['Symbol', 'PDBresNum', 'restp', 'x', 'y', 'z']]
                df_cur = df_tmp.groupby('PDBresNum').mean().reset_index()
                df_cur['resnm'] = [AAdict2[int(i)] for i in df_cur['restp']]
                allres = set(df_cur['resnm'])
                x1, y1, z1 = df_cur['x'].tolist(), df_cur['y'].tolist(), df_cur['z'].tolist()
                if bkbn:
                    ax.plot3D(xs = x1, ys = y1, zs = z1, color = 'black')
                if clr is None:
                    for curres in allres:     
                        df_cur_fil = df_cur.loc[df_cur['resnm'] == curres]
                        x, y, z = df_cur_fil['x'].tolist(), df_cur_fil['y'].tolist(), df_cur_fil['z'].tolist()
                        ax.scatter3D(xs = x, ys = y, zs = z, label = curres, s = 50)
                    if len(allres) > 7:
                        ax.legend(loc="lower left", mode = "expand", ncol = 7)  
                    else:
                        ax.legend(loc="best")  
                else:
                    ax.scatter3D(xs = x1, ys = y1, zs = z1, color = clr)            
            ax.set_title('Display of residues (PDB ID: 1F5L)')

    def Plot_surf(self, clr = 'crimson', heavyAtm = True, alpha = 3):
        """
        Plot surface of a protein using 3d alpha shape modeling
        Parameters:
            clr - Color for plots
            heavyAtm - Wether to use heave atoms only (= True) or not (= False)
            alpha - Parameter for alpha shape modeling
        """
        df = self.ProAtmProp if not heavyAtm else self.ProAtmProp.loc[(self.ProAtmProp['AtomicNum'] > 1) & (self.ProAtmProp['PDBresName'] != 'HOH')]
#        df = ProAtmProp if not heavyAtm else ProAtmProp.loc[(ProAtmProp['AtomicNum'] > 1) & (ProAtmProp['PDBresName'] != 'HOH')]
        
        all_pos = np.array(df['xyz'].to_list())
        x, y, z, tri_idx = alpha_shape_plot_prepare(all_pos = all_pos, alpha = alpha)
        
        plotclr = clr if clr is not None else 'gold'
        ax = plt.figure(figsize = (12, 12)).gca(projection = '3d')
        ax.plot_trisurf(x, y, z, triangles = tri_idx, color = plotclr)
        ax.set_title('Display of protein surface (PDB ID: 1F5L, alpha = %.1f)' % alpha)
        plt.show()
    
    def Plot_interface(self, clr = 'gold', heavyAtm = True, alpha = None, cutoff = 3):
        """
        Plot the interface on a protein for a binding ligand
        Parameters:
            clr - Color for plots
            heavyAtm - Wether to use heave atoms only (= True) or not (= False)
            alpha - Parameter for alpha shape modeling (method 1, if None use method 2)
            cutoff - Parameter for extracting contacting atoms (method 2, if None use method 1)
        """
        df_pro = self.ProAtmProp if not heavyAtm else self.ProAtmProp.loc[(self.ProAtmProp['AtomicNum'] > 1) & (self.ProAtmProp['PDBresName'] != 'HOH')]
#        df_pro = ProAtmProp if not heavyAtm else ProAtmProp.loc[(ProAtmProp['AtomicNum'] > 1) & (ProAtmProp['PDBresName'] != 'HOH')]
        contacts = np.nonzero(self.pd < cutoff)
        conts = (list(set(contacts[0])), list(set(contacts[1])))      
        
        if alpha is not None:
            coor_alpha_pro = np.array(df_pro['xyz'].tolist())
            coor_alpha_comp = np.append(coor_alpha_pro, self.lig[0], axis = 0)
#            x1, y1, z1, tri_idx1 = alpha_shape_plot_prepare(all_pos = coor_alpha_pro, alpha = alpha)
#            x2, y2, z2, tri_idx2 = alpha_shape_plot_prepare(all_pos = coor_alpha_comp, alpha = alpha)
            
            vers1, edges1, tris1 = alpha_shape_3D(pos = coor_alpha_pro, alpha = alpha)
            vers2, edges2, tris2 = alpha_shape_3D(pos = coor_alpha_comp, alpha = alpha)
            intf = [i for i in tris1.tolist() if i not in tris2.tolist()]
    
            tricoor = np.take(coor_alpha_pro, intf, axis = 0)
            tri_points = list(chain.from_iterable(tricoor))
            x, y, z = zip(*tri_points)
            tri_idx = [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(intf))]  
            
            ax = plt.figure(figsize = (12, 12)).gca(projection = '3d')
            ax.plot_trisurf(x, y, z, triangles = tri_idx, color = clr)
            ax.set_title('Display of protein-binding interface using alpha shape modeling \n(PDB ID: 1F5L, alpha = %.1f)' % alpha)
            plt.show()

        else:
            if cutoff is None or cutoff < 0:
                print('Wrong parameters!')
            else:
                coor_pro = self.pro[0][conts[0]] 
                x, y, z = [item[0] for item in coor_pro.tolist()], [item[1] for item in coor_pro.tolist()], [item[2] for item in coor_pro.tolist()]
                            
                fig = plt.figure(figsize = (12, 12))
                ax = plt.axes(projection = '3d')   
                ax.scatter3D(xs = x, ys = y, zs = z, marker = 'o', c = clr)
                ax.set_title('Display of protein-binding interface (PDB ID: 1F5L, cutoff = %.1f A)' % cutoff)
                plt.show()


            




    
