from predict import classify_news


def main():
    news_article = input("Please enter the news article: ")

    prediction = classify_news(news_article)

    print(f"This news is {prediction}")


if __name__ == "__main__":
    main()
