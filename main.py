import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import time
import matplotlib.pyplot as plt
from itertools import combinations
import os

class HybridApriori:
    def __init__(self, min_support=0.01, min_confidence=0.5, max_length=None):
        """
        Initialize the Hybrid Apriori algorithm with parameters.
        
        Parameters:
        -----------
        min_support : float, default=0.01
            The minimum support threshold for frequent itemsets
        min_confidence : float, default=0.5
            The minimum confidence threshold for association rules
        max_length : int, default=None
            The maximum length of itemsets to consider
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_length = max_length
        self.frequent_itemsets = {}
        self.transaction_count = 0
        self.item_counts = defaultdict(int)
        self.rules = []
    
    def fit(self, transactions):
        """
        Find frequent itemsets and association rules in the transactions.
        
        Parameters:
        -----------
        transactions : list of lists
            List of transactions, where each transaction is a list of items
        
        Returns:
        --------
        self : object
        """
        start_time = time.time()
        self.transaction_count = len(transactions)
        min_support_count = self.min_support * self.transaction_count
        
        print(f"Minimum support count: {min_support_count:.2f} transactions")
        
        # Convert transactions to sets for faster processing
        transaction_sets = [set(transaction) for transaction in transactions]
        
        # Count individual items (1-itemsets) using Counter for efficiency
        item_counter = Counter()
        for transaction in transactions:
            item_counter.update(transaction)
        
        self.item_counts = dict(item_counter)
        
        # Filter items with minimum support
        frequent_items = {item: count for item, count in self.item_counts.items() 
                         if count >= min_support_count}
        
        if not frequent_items:
            print("No frequent items found with the given minimum support.")
            return self
        
        print(f"Found {len(frequent_items)} frequent items")
        
        # Store frequent 1-itemsets
        for item, count in frequent_items.items():
            self.frequent_itemsets[(item,)] = count
        
        # Sort frequent items by frequency (descending) for optimization
        sorted_items = sorted(frequent_items.keys(), key=lambda x: frequent_items[x], reverse=True)
        
        # Find frequent 2-itemsets
        print("Finding frequent 2-itemsets...")
        t0 = time.time()
        
        frequent_2_itemsets = {}
        for i, item1 in enumerate(sorted_items):
            # Create a list to count co-occurrences for this item with all other items
            item_co_occurrences = defaultdict(int)
            
            # Count co-occurrences in transactions
            for transaction_set in transaction_sets:
                if item1 in transaction_set:
                    for item2 in transaction_set:
                        if item2 in frequent_items and item1 != item2:
                            if frequent_items[item1] > frequent_items[item2]:
                                item_co_occurrences[item2] += 1
            
            # Filter by minimum support
            for item2, count in item_co_occurrences.items():
                if count >= min_support_count:
                    # Ensure items are ordered by frequency for consistent keys
                    if frequent_items[item1] > frequent_items[item2]:
                        frequent_2_itemsets[(item1, item2)] = count
                    else:
                        frequent_2_itemsets[(item2, item1)] = count
        
        print(f"Found {len(frequent_2_itemsets)} frequent 2-itemsets in {time.time() - t0:.2f} seconds")
        
        # Update frequent itemsets with 2-itemsets
        self.frequent_itemsets.update(frequent_2_itemsets)
        
        # If max_length is 2 or no 2-itemsets found, stop here
        if self.max_length == 2 or not frequent_2_itemsets:
            if len(self.frequent_itemsets) > 1:
                t0 = time.time()
                self._generate_rules()
                print(f"Generated {len(self.rules)} association rules in {time.time() - t0:.2f} seconds")
            
            print(f"Mining completed in {time.time() - start_time:.2f} seconds")
            print(f"Found {len(self.frequent_itemsets)} frequent itemsets")
            print(f"Generated {len(self.rules)} association rules")
            return self
        
        # Find frequent 3-itemsets (if needed)
        if self.max_length >= 3:
            print("Finding frequent 3-itemsets...")
            t0 = time.time()
            
            frequent_3_itemsets = {}
            # Get all 2-itemsets
            two_itemsets = list(frequent_2_itemsets.keys())
            
            # Create a dictionary to quickly look up if a 2-itemset is frequent
            two_itemset_dict = {frozenset(itemset): count for itemset, count in frequent_2_itemsets.items()}
            
            # For each transaction, find all possible 3-itemsets and check if all their subsets are frequent
            three_itemset_counter = Counter()
            
            for transaction_set in transaction_sets:
                # Only consider items that are in frequent_items
                transaction_frequent_items = [item for item in transaction_set if item in frequent_items]
                
                # If transaction has at least 3 frequent items
                if len(transaction_frequent_items) >= 3:
                    # Generate all possible 3-itemsets from this transaction
                    for three_itemset in combinations(transaction_frequent_items, 3):
                        # Check if all 2-itemset subsets are frequent
                        is_valid = True
                        for two_subset in combinations(three_itemset, 2):
                            if frozenset(two_subset) not in two_itemset_dict:
                                is_valid = False
                                break
                        
                        if is_valid:
                            three_itemset_counter[three_itemset] += 1
            
            # Filter by minimum support
            for itemset, count in three_itemset_counter.items():
                if count >= min_support_count:
                    frequent_3_itemsets[itemset] = count
            
            print(f"Found {len(frequent_3_itemsets)} frequent 3-itemsets in {time.time() - t0:.2f} seconds")
            
            # Update frequent itemsets with 3-itemsets
            self.frequent_itemsets.update(frequent_3_itemsets)
        
        # Generate association rules
        if len(self.frequent_itemsets) > 1:
            t0 = time.time()
            self._generate_rules()
            print(f"Generated {len(self.rules)} association rules in {time.time() - t0:.2f} seconds")
        
        print(f"Mining completed in {time.time() - start_time:.2f} seconds")
        print(f"Found {len(self.frequent_itemsets)} frequent itemsets")
        print(f"Generated {len(self.rules)} association rules")
        
        return self
    
    def _get_support(self, itemset):
        """Calculate support for an itemset"""
        if len(itemset) == 1:
            return self.item_counts.get(itemset[0], 0) / self.transaction_count
        else:
            return self.frequent_itemsets.get(itemset, 0) / self.transaction_count
    
    def _generate_rules(self):
        """Generate association rules from frequent itemsets"""
        self.rules = []
        
        for itemset, support_count in self.frequent_itemsets.items():
            if len(itemset) < 2:
                continue
                
            # Calculate support for the itemset
            itemset_support = support_count / self.transaction_count
            
            # Generate all non-empty subsets except the itemset itself
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    # Calculate the consequent (items in itemset but not in antecedent)
                    consequent = tuple(item for item in itemset if item not in antecedent)
                    
                    # Get antecedent support
                    if len(antecedent) == 1:
                        antecedent_support = self.item_counts.get(antecedent[0], 0) / self.transaction_count
                    else:
                        antecedent_support = self.frequent_itemsets.get(antecedent, 0) / self.transaction_count
                    
                    if antecedent_support > 0:
                        # Calculate confidence
                        confidence = itemset_support / antecedent_support
                        
                        # If confidence meets threshold, add the rule
                        if confidence >= self.min_confidence:
                            # Calculate lift
                            consequent_support = self._get_support(consequent)
                            if consequent_support > 0:
                                lift = confidence / consequent_support
                                
                                self.rules.append({
                                    'antecedent': antecedent,
                                    'consequent': consequent,
                                    'support': itemset_support,
                                    'confidence': confidence,
                                    'lift': lift
                                })
        
        # Sort rules by confidence (descending)
        self.rules.sort(key=lambda x: x['confidence'], reverse=True)
    
    def get_top_rules(self, n=10):
        """Return the top n rules sorted by confidence"""
        return self.rules[:n]
    
    def print_itemsets(self, min_length=2, max_length=None):
        """Print frequent itemsets within specified length range"""
        sorted_itemsets = sorted(
            [(itemset, count / self.transaction_count) 
             for itemset, count in self.frequent_itemsets.items()
             if max_length is None or len(itemset) <= max_length
             if len(itemset) >= min_length],
            key=lambda x: x[1], 
            reverse=True
        )
        
        print(f"\nTop Frequent Itemsets (min_length={min_length}, max_length={max_length}):")
        print("=" * 50)
        print(f"{'Itemset':<30} {'Support':<10}")
        print("-" * 50)
        
        for itemset, support in sorted_itemsets[:20]:  # Show top 20
            print(f"{str(itemset):<30} {support:.4f}")
    
    def print_rules(self, n=10):
        """Print top n association rules"""
        print(f"\nTop {n} Association Rules:")
        print("=" * 80)
        print(f"{'Antecedent':<25} {'Consequent':<25} {'Support':<10} {'Confidence':<10} {'Lift':<10}")
        print("-" * 80)
        
        for rule in self.rules[:n]:
            print(f"{str(rule['antecedent']):<25} {str(rule['consequent']):<25} "
                  f"{rule['support']:.4f} {rule['confidence']:.4f} {rule['lift']:.4f}")
    
    def plot_support_distribution(self):
        """Plot the distribution of support values for frequent itemsets"""
        supports = [count / self.transaction_count for count in self.frequent_itemsets.values()]
        
        plt.figure(figsize=(10, 6))
        plt.hist(supports, bins=20, alpha=0.7)
        plt.title('Distribution of Support Values')
        plt.xlabel('Support')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig('support_distribution.png')
        plt.close()
        
        print("Support distribution plot saved as 'support_distribution.png'")
    
    def plot_top_items(self, n=20):
        """Plot the top n most frequent individual items"""
        # Get the top n items by support
        top_items = sorted(
            [(item, count / self.transaction_count) 
             for item, count in self.item_counts.items()],
            key=lambda x: x[1], 
            reverse=True
        )[:n]
        
        items, supports = zip(*top_items)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(items)), supports, alpha=0.7)
        plt.xticks(range(len(items)), items, rotation=45)
        plt.title(f'Top {n} Most Frequent Items')
        plt.xlabel('Item ID')
        plt.ylabel('Support')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('top_items.png')
        plt.close()
        
        print(f"Top {n} items plot saved as 'top_items.png'")
    
    def plot_top_rules(self, n=10):
        """Plot the top n association rules by lift"""
        if not self.rules or n <= 0:
            print("No rules to plot or invalid n value.")
            return
        
        # Get top n rules by lift
        top_rules = sorted(self.rules, key=lambda x: x['lift'], reverse=True)[:n]
        
        # Create labels and values for the plot
        labels = [f"{rule['antecedent']} → {rule['consequent']}" for rule in top_rules]
        lifts = [rule['lift'] for rule in top_rules]
        confidences = [rule['confidence'] for rule in top_rules]
        supports = [rule['support'] * 100 for rule in top_rules]  # Convert to percentage
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot lift values
        bars1 = ax1.barh(range(len(labels)), lifts, alpha=0.7, color='skyblue')
        ax1.set_yticks(range(len(labels)))
        ax1.set_yticklabels(labels)
        ax1.set_xlabel('Lift')
        ax1.set_title('Top Rules by Lift')
        ax1.grid(True, alpha=0.3)
        
        # Add lift values at the end of each bar
        for i, bar in enumerate(bars1):
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{lifts[i]:.2f}', va='center')
        
        # Plot confidence and support values
        x = np.arange(len(labels))
        width = 0.35
        bars2 = ax2.barh(x - width/2, confidences, width, alpha=0.7, color='lightgreen', label='Confidence')
        bars3 = ax2.barh(x + width/2, supports, width, alpha=0.7, color='salmon', label='Support (%)')
        
        ax2.set_yticks(x)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Value')
        ax2.set_title('Confidence and Support for Top Rules')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add values at the end of each bar
        for i, bar in enumerate(bars2):
            ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{confidences[i]:.2f}', va='center')
        
        for i, bar in enumerate(bars3):
            ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{supports[i]:.3f}%', va='center')
        
        plt.tight_layout()
        plt.savefig('top_rules.png')
        plt.close()
        
        print("Top rules plot saved as 'top_rules.png'")
    
    def analyze_item_relationships(self, item_id=None, top_n=5):
        """
        Analyze relationships for a specific item or find the most connected items
        
        Parameters:
        -----------
        item_id : int, default=None
            The item ID to analyze. If None, will find the most connected items.
        top_n : int, default=5
            Number of top relationships to show
        """
        if not self.frequent_itemsets:
            print("No frequent itemsets found.")
            return
        
        if item_id is not None:
            # Find all itemsets containing the specified item
            related_itemsets = []
            for itemset, count in self.frequent_itemsets.items():
                if len(itemset) > 1 and item_id in itemset:
                    related_itemsets.append((itemset, count))
            
            if not related_itemsets:
                print(f"No relationships found for item {item_id}.")
                return
            
            # Sort by support (count)
            related_itemsets.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nTop {min(top_n, len(related_itemsets))} relationships for item {item_id}:")
            print("=" * 50)
            print(f"{'Related Items':<30} {'Support':<10} {'Count':<10}")
            print("-" * 50)
            
            for itemset, count in related_itemsets[:top_n]:
                # Get the related items (excluding the item_id)
                related_items = tuple(item for item in itemset if item != item_id)
                print(f"{str(related_items):<30} {count/self.transaction_count:.4f} {count:<10}")
        else:
            # Find the most connected items (items that appear in the most itemsets)
            item_connections = Counter()
            for itemset in self.frequent_itemsets:
                if len(itemset) > 1:  # Only consider itemsets with at least 2 items
                    for item in itemset:
                        item_connections[item] += 1
            
            # Get the top connected items
            top_connected = item_connections.most_common(top_n)
            
            if not top_connected:
                print("No connected items found.")
                return
            
            print(f"\nTop {len(top_connected)} most connected items:")
            print("=" * 50)
            print(f"{'Item':<10} {'Connections':<15} {'Frequency':<15}")
            print("-" * 50)
            
            for item, connections in top_connected:
                item_frequency = self.item_counts.get(item, 0)
                print(f"{item:<10} {connections:<15} {item_frequency:<15}")

def load_transactions(file_path):
    """Load transactions from a file where each line is a transaction"""
    transactions = []
    with open(file_path, 'r') as f:
        for line in f:
            # Split the line by spaces and convert to integers
            items = [int(item) for item in line.strip().split()]
            if items:  # Only add non-empty transactions
                transactions.append(items)
    return transactions

def save_results_to_csv(apriori):
    """Save the results to CSV files for further analysis"""
    # Save frequent itemsets
    itemsets_data = []
    for itemset, count in apriori.frequent_itemsets.items():
        if len(itemset) >= 2:  # Only include itemsets with 2 or more items
            support = count / apriori.transaction_count
            itemsets_data.append({
                'itemset': str(itemset),
                'length': len(itemset),
                'support': support,
                'count': count
            })
    
    if itemsets_data:
        pd.DataFrame(itemsets_data).sort_values(by='support', ascending=False).to_csv(
            'frequent_itemsets.csv', index=False)
        print("Frequent itemsets saved to 'frequent_itemsets.csv'")
    
    # Save association rules
    if apriori.rules:
        rules_data = []
        for rule in apriori.rules:
            rules_data.append({
                'antecedent': str(rule['antecedent']),
                'consequent': str(rule['consequent']),
                'support': rule['support'],
                'confidence': rule['confidence'],
                'lift': rule['lift']
            })
        
        pd.DataFrame(rules_data).to_csv('association_rules.csv', index=False)
        print("Association rules saved to 'association_rules.csv'")

def main():
    # Load transactions from file
    file_path = 'Sales1998.txt'
    transactions = load_transactions(file_path)
    
    print(f"Loaded {len(transactions)} transactions")
    print(f"Average transaction length: {sum(len(t) for t in transactions) / len(transactions):.2f} items")
    
    # Get unique items count
    unique_items = set()
    for transaction in transactions:
        unique_items.update(transaction)
    print(f"Dataset contains {len(unique_items)} unique items")
    
    # Analyze item frequency distribution
    item_counter = Counter()
    for transaction in transactions:
        item_counter.update(transaction)
    
    # Find top items
    top_items = item_counter.most_common(20)
    print("\nTop 20 most frequent items:")
    for item, count in top_items:
        print(f"Item {item}: {count} occurrences ({count/len(transactions)*100:.2f}%)")
    
    # For this extremely sparse dataset, use absolute count instead of percentage
    # The most frequent item only appears in 0.42% of transactions
    min_support_count = 10  # Items that appear together at least 10 times
    min_support = min_support_count / len(transactions)
    print(f"\nUsing absolute minimum support count: {min_support_count} transactions ({min_support*100:.4f}%)")
    
    # Initialize and run Hybrid Apriori algorithm with appropriate parameters
    apriori = HybridApriori(min_support=min_support, min_confidence=0.05, max_length=3)
    apriori.fit(transactions)
    
    # Print results
    apriori.print_itemsets(min_length=2, max_length=3)
    apriori.print_rules(n=15)
    
    # Generate visualizations
    apriori.plot_support_distribution()
    apriori.plot_top_items(n=20)
    apriori.plot_top_rules(n=10)
    
    # Analyze item relationships
    apriori.analyze_item_relationships(top_n=10)
    
    # Analyze specific items (top 3 most frequent)
    if top_items:
        for item, _ in top_items[:3]:
            apriori.analyze_item_relationships(item_id=item, top_n=5)
    
    # Save results to CSV files for further analysis
    save_results_to_csv(apriori)

if __name__ == "__main__":
    main()