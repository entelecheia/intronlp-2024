# Session 1 - Protein Structure Prediction Using Artificial Intelligence

This session will cover the advances in protein structure prediction that led to the 2024 Nobel Prize in Chemistry, awarded to Demis Hassabis and John Jumper for their groundbreaking work with **AlphaFold2**. We will explore the history of protein folding challenges, the development of AlphaFold, and its implications for the scientific community.

## Introduction to Protein Structure and Its Importance

- **Proteins as Chemical Tools**: Proteins are essential for nearly all biological processes, functioning as enzymes, hormones, antibodies, and structural components.
  - **Amino Acids as Building Blocks**: Proteins are made up of 20 different amino acids. The sequence of these amino acids determines the **three-dimensional structure** of the protein, which in turn defines its function.
- **Historical Methods of Protein Structure Determination**:
  - **X-ray Crystallography**: Since the 1950s, X-ray crystallography has been the primary method for determining protein structures, but it is a labor-intensive and time-consuming process.
  - **Challenges in Predicting Structure from Sequence**: The challenge of predicting a protein's structure from its amino acid sequence has been an enduring problem in biochemistry since the 1960s, due to the vast number of possible conformations each protein can adopt (known as **Levinthal's paradox**).

## The Rise of AlphaFold and AI in Protein Folding

- **Critical Assessment of Protein Structure Prediction (CASP)**: In 1994, the **CASP competition** was established to encourage progress in predicting protein structures from amino acid sequences. For decades, the improvements were incremental, with no major breakthroughs until AI entered the scene.
- **DeepMind's Contribution**:
  - **Demis Hassabis and AlphaFold**: In 2018, Demis Hassabis, co-founder of DeepMind, and his team entered **AlphaFold** in the CASP competition. AlphaFold used **deep learning** to achieve unprecedented accuracy in structure prediction, marking a breakthrough moment.
  - **AlphaFold2 and the 2020 CASP Competition**: **AlphaFold2**, an improved version, revolutionized the field by using **transformer neural networks** to predict protein structures with near-experimental accuracy, solving a problem that had persisted for over 50 years.
  - **Neural Networks and Transformers**: AlphaFold2 relies on **transformer architecture**, which enables the model to effectively find relationships in amino acid sequences and predict their folding patterns. This approach outperformed traditional computational methods.

## How AlphaFold2 Works

- **Data Entry and Sequence Analysis**: AlphaFold2 takes an amino acid sequence as input and searches for similar sequences in existing databases. It aligns sequences to identify regions that have evolved together, which helps predict the folding patterns.
  - ![](figs/fig2_ke_en_24.jpeg)
- **Distance Maps and Neural Network Analysis**: The AI then creates a **distance map** that shows how closely amino acids are positioned to each other in the final 3D structure.
  - **Transformer Neural Networks**: Using transformers, AlphaFold2 identifies the most important elements of the protein to focus on, refining the predicted structure through iterative cycles until it arrives at a high-confidence model.

## Impact and Applications

- **Massive Acceleration of Protein Research**: Prior to AlphaFold2, determining a single protein structure could take years. With AlphaFold2, predictions can be made in **minutes**, with accuracy comparable to experimental methods like X-ray crystallography.
- **Widespread Adoption**: The code for AlphaFold2 has been made publicly available, and by 2024, it had been used by over two million researchers worldwide. The ability to rapidly predict protein structures has accelerated advances in drug discovery, understanding genetic diseases, and developing new biotechnologies.
- **Case Studies**:
  - **Drug Development**: AlphaFold2 has helped pharmaceutical companies identify drug targets by providing structural insights into proteins involved in diseases.
  - **Environmental Applications**: Understanding the structure of enzymes that can break down plastics has paved the way for innovations in **plastic recycling**.
  - ![](figs/fig5_ke_en_24.jpeg)

## Key Takeaway

- **AI and Biochemistry**: The use of **deep learning** and **transformer models** by Demis Hassabis and John Jumper has fundamentally transformed the field of biochemistry by solving a problem that had been a significant barrier for decades.
- **Broad Implications**: The applications of AlphaFold2 go beyond academic research and are already impacting areas like **medicine**, **agriculture**, and **environmental science**.
- **Collaboration and Accessibility**: Making AlphaFold2 publicly accessible has democratized protein research, allowing scientists from all disciplines to leverage this powerful tool.
