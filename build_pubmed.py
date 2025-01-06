from src.datasets import PubMed

def main():
    pm = PubMed()
    
    print('Collecting abstracts...')
    pm.collect_abstracts()
    
    print('Generating questions...')
    pm.generate_questions(num_questions=250)
    
    print('Done!')

if __name__ == "__main__":
    main()
