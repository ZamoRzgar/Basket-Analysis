# Market Basket Analysis using Hybrid Apriori Algorithm

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Project Overview
This project implements a Hybrid Apriori algorithm for market basket analysis, designed specifically to handle extremely sparse transaction datasets. The implementation includes optimizations for efficient processing and comprehensive analysis tools for interpreting the results.

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="400">
</p>

## Dataset
The analysis uses the Sales1998.txt dataset with the following characteristics:
- 34,070 transactions with an average of 4.83 items per transaction
- 1,559 unique items
- Extremely sparse: the most frequent item (277) appears in only 0.42% of transactions

## Algorithm Implementation
The `HybridApriori` class implements a modified version of the classic Apriori algorithm with several optimizations:

1. **Efficient Data Structures**:
   - Uses sets for transactions to speed up item lookups
   - Employs Counter for efficient item counting
   - Utilizes defaultdict for counting co-occurrences

2. **Frequency-Based Sorting**:
   - Sorts items by frequency in descending order to optimize the search process

3. **Pruning Strategies**:
   - Applies the Apriori principle to prune candidate itemsets
   - Implements early termination if no frequent itemsets are found at a certain level

4. **Memory Efficiency**:
   - Processes transactions in a streaming fashion
   - Only stores frequent itemsets, not all candidate itemsets

## Key Findings
Due to the extreme sparsity of the dataset, the analysis required using an absolute support count (10 transactions, equivalent to 0.0294%) instead of a percentage-based threshold. This approach yielded:

- 6 frequent 2-itemsets
- 12 association rules with high lift values (ranging from 23 to 32)
- Item 132 appearing in 2 different frequent itemsets, making it the most connected item
- Interestingly, the top 3 most frequent items (277, 1352, 846) don't appear in any frequent itemsets

## Files in this Repository
- `main.py`: Implementation of the Hybrid Apriori algorithm and analysis tools
- `Sales1998.txt`: Raw transaction data (each line represents a transaction)
- `frequent_itemsets.csv`: Output file containing discovered frequent itemsets
- `association_rules.csv`: Output file containing generated association rules
- `requirements.txt`: List of Python dependencies

## Installation & Setup

### Prerequisites
- Python 3.13 or higher
- Required packages: numpy, pandas, matplotlib

### Installation
1. Clone this repository:
```bash
git clone https://github.com/ZamoRzgar/basket-analysis.git
cd basket-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
To run the analysis:
```bash
python main.py
```

The script will:
1. Load and analyze the transaction data
2. Apply the Hybrid Apriori algorithm with appropriate parameters
3. Generate and display frequent itemsets and association rules
4. Create visualizations of the results
5. Save the findings to CSV files

## Parameters
The algorithm uses the following default parameters:
- Minimum support count: 10 transactions (0.0294%)
- Minimum confidence: 0.05 (5%)
- Maximum itemset length: 3

These parameters can be adjusted in the `main()` function to suit different analysis needs.

## Visualizations
The implementation includes several visualization tools:
- Support distribution for frequent itemsets
- Top frequent items
- Top association rules by lift
- Item relationship networks

## Results
The analysis reveals strong associations between specific item pairs, with lift values indicating that these associations are 23-32 times stronger than would be expected by random chance. These insights can be valuable for:
- Product placement strategies
- Targeted promotions and cross-selling
- Inventory management
- Customer behavior understanding

## Future Work
Potential extensions to this project include:
- Testing different support thresholds to balance pattern discovery and statistical significance
- Implementing alternative algorithms like FP-Growth or Eclat for comparison
- Incorporating temporal aspects to analyze how associations change over time
- Applying clustering techniques to group similar items before running association rule mining

## References
1. Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. In Proceedings of the 20th International Conference on Very Large Data Bases (VLDB), 487-499.
2. Han, J., Pei, J., & Kamber, M. (2011). Data Mining: Concepts and Techniques (3rd ed.). Morgan Kaufmann.
3. Tan, P. N., Steinbach, M., & Kumar, V. (2005). Introduction to Data Mining. Addison-Wesley.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
Zamo Rzgar Ahmed

---

<p align="center">
  <i>If you found this project helpful, please consider giving it a star ⭐</i>
</p>
