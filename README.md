# TGC-ARG

The increasing severity of antibiotic resistance poses significant challenges to everyday life, agriculture, and clinical medical treatments. Although various methods have been developed to study antibiotic resistance (ARG), such as culture-based methods and whole-genome sequencing, they are often time-consuming, labor-intensive, and have relatively low accuracy. Additionally, existing datasets are often scattered, making it challenging to comprehensively analyze antibiotic resistance gene sequences. To improve ARG prediction, this paper presents a new publicly available multi-label dataset called ARSS (Antibiotic Resistance Sequence Statistics) that can be used for subsequent ARG prediction and validation. The paper proposes a deep learning framework based on Transformer and GRU (Gated Recurrent Unit) to effectively extract resistance features from protein sequences and secondary structures. Furthermore, a siamese network based on contrastive learning is utilized to learn more discriminative feature representations from the similarity and dissimilarity between samples. The results demonstrate that TGC-ARG outperforms existing state-of-the-art methods, validating its robustness and effectiveness.

## Dataset
### ARSS.fasta
Antibiotic resistance sequence statistics，and the data format was as follows:
```
>beta-lactam|antibiotic target protection|0
MAEPVLSVKDLDIRFTTPDGNVHAVKKVSFDIAPGECLGVVGESGSGKSQLFMACIGLLAGNGKATGSVTYRGQELLGQPAAKLNAIRGAKITMIFQDPLTSLTPHMRIGDQIVESLRTHSKLSKGEAEKRAIQALELVRIPEAKRRMRQYPHELSGGMRQRVMIAMATACGPDLLIADEPTTALDVTVQAQILDIMRDLRKELGTSIALISHDMGVIASICDRVQVMRYGEFVETGPADDIFYHPQHPYTRMLLEAMPRIDQPVREGRAALKPLAPQEARTTLLEVDNVKVHFPIQMGGVFFGKYKPLRAVDGVSFTLHQGETIGIVGESGCGKSTLARAVLELLPKTTGGVVWMGRDLGALPPAELRRARKDFQIVFQDPLASLDPRMTIGQSIAEPLQSLEPELSKHEVQSRVRAIMEKVGLDPDWINRYPHEFSGGQNQRVGIARAMILKPKLIVCDEAVSALDVSIQAQIVDLILSLQAEFGMSIIFISHDLSVVRQVSHRVMVLYLGRVVELASRDAIYEDARHPYTKALISAVTVPDPRAERLKKRRELPGELPSPLDTRSALMFLKSKRIDDPDAEQYVPKLIEVAPGHFVAEHDPFEVVEMTG
``` 


### ARSS-90-seq+stru.csv
Input dataset containing sequence information, labeling information, and secondary structure information，and the data format was as follows:
```
MAEPVLSVKDLDIRFTTPDGNVHAVKKVSFDIAPGECLGVVGESGSGKSQLFMACIGLLAGNGKATGSVTYRGQELLGQPAAKLNAIRGAKITMIFQDPLTSLTPHMRIGDQIVESLRTHSKLSKGEAEKRAIQALELVRIPEAKRRMRQYPHELSGGMRQRVMIAMATACGPDLLIADEPTTALDVTVQAQILDIMRDLRKELGTSIALISHDMGVIASICDRVQVMRYGEFVETGPADDIFYHPQHPYTRMLLEAMPRIDQPVREGRAALKPLAPQEARTTLLEVDNVKVHFPIQMGGVFFGKYKPLRAVDGVSFTLHQGETIGIVGESGCGKSTLARAVLELLPKTTGGVVWMGRDLGALPPAELRRARKDFQIVFQDPLASLDPRMTIGQSIAEPLQSLEPELSKHEVQSRVRAIMEKVGLDPDWINRYPHEFSGGQNQRVGIARAMILKPKLIVCDEAVSALDVSIQAQIVDLILSLQAEFGMSIIFISHDLSVVRQVSHRVMVLYLGRVVELASRDAIYEDARHPYTKALISAVTVPDPRAERLKKRRELPGELPSPLDTRSALMFLKSKRIDDPDAEQYVPKLIEVAPGHFVAEHDPFEVVEMTG,0,CCCCEEEEEEEEEEEECCCCEEEEEEEEEEEECCCEEEEEEEECCCECEEEEEECCCCECCCEEEEEEEEECCEEECCCCHHHHHHHCCCCEEEEEECCCCCCCCCCCHHHHHHHHHHHCCCCCHHHHHHHHHHHHHHCCCCCHHHHCCCCHHHHHHHHHHHHHHHHHHHHCCCEEECCCCCCCCCHHHHHHHHHHHHHHHHHHCCEEEEECHHHHHHHHHCCEEEEECCCEEEEEEEHHHHHCCCCCHHHHHHHHCCCCCCCCCCCCCCCCCCCCCCCCCCEEEEEECEEEEEEECCCCCCCCCCEEEEECCCCCEEEECCCEEEEEEECCCCHHHHHHHHHCCCCCCCCEEEECCEEHHHCCHHHHHHCCCCEEEEECCHHHHCCCCCCHHHHHHHHHHHHCCCCCHHHHHHHHHHHHHHHCCCHHHCCCCCCCCCCCHHHHHHHHHHHHCCCCEEEEHHHHHHHHHHHHHHHHHHHHHHHHHHCCEEEEECCCHHHHHHCCCEEEEECCCEEEEEECHHHHCCCCCCHHHHHHHCCCCCCCHHHHCCCCCECCCCCCCCCCCCHHHHHHHCCCCCCCCCCCCCCCCEEEECCCCEEECCCHHHHHHHCC
```

## Usage
```
pip install -r requirements.txt
python main.py
```

