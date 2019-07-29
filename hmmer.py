# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 20:05:43 2019

@author: falcon1
"""
import os
from subprocess import run
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio.SeqRecord import SeqRecord
#######
# get a protein's homology proteins by HMMER
# evalue = -1, get the top 10 (if has) homology proteins
# evalue != -1, get the proteins whoes E-value < evalue
######
def getHomoProteinsByHMMER(seqrecord, evalue=-1):
    # HMMER command line
    hmmbuildCMD = r'hmmbuild input.hmm input.fasta' 
    hmmsearchCMD = r'hmmsearch input.hmm e:/uniprot_sprot.fasta > input.out'
    if os.path.exists('input.hmm'):
        os.remove('input.hmm')
    if os.path.exists('input.fasta'):
        os.remove('input.fasta')
    if os.path.exists('input.out'):
        os.remove('input.out')
        
    fpw = open('input.fasta','w') 
    SeqIO.write(seqrecord, fpw, 'fasta')
    fpw.close()
    
    #os.system(hmmbuildCMD)
    #os.system(hmmsearchCMD)
    run(hmmbuildCMD, shell=True)
    run(hmmsearchCMD, shell=True)
    
    homoProtein=[]
    fpr = open('input.out','r')
    lines = fpr.readlines()
    fpr.close()
    
    i = 14
    if evalue < 0:
        while i <= 24:
            line = lines[i]
            if line.strip() == "":
                break
            if '----' in line:
                break
            s = line.split()
            homoProtein.append(s[8])
            i = i + 1
    else:
        while True:
            line = lines[i]
            if line.strip() == "":
                break
            if '----' in line:
                break
            s = line.split()
            if float(s[0]) > evalue:
                break
            homoProtein.append(s[8])
            i = i + 1
                
    return homoProtein

# example of usage
def main():
    seqdata ='SLFEQLGGQAAVQAVTAQFYANIQADATVATFFNGIDMPNQTNKTAAFLCAALGGPNAWTGRNLKEVHAN\
MGVSNAQFTTVIGHLRSALTGAGVAAALVEQTVAVAETVRGDVVTV'
    head = 'd1d1wa'
    seq = Seq(seqdata, IUPAC.ExtendedIUPACProtein)
    seqrecord = SeqRecord(seq, id=head)
    homoProtein = getHomoProteinsByHMMER(seqrecord)
    print(homoProtein)
    
if __name__ == '__main__':
    main()
    