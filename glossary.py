from explain import explain_term

TERMS = ["Narrative","FinBERT","Price score","Volume Z","Gap","Threshold","Confidence","Action","Reason"]

def main():
    print("Glossary:")
    for t in TERMS:
        print(f"- {t}: {explain_term(t)}")

if __name__ == "__main__":
    main()
