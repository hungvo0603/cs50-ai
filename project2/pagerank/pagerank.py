from mmap import PAGESIZE
import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

 
def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    result = {}
    links = corpus[page]
    if len(links) == 0:
        keys = corpus.keys()
        for key in keys:
            result[key] = float(1/len(keys))
    else:
        for link in links:
            result[link] = float(damping_factor/len(links)) + float((1-damping_factor)/(len(links)+1))
        result[page] = float((1-damping_factor)/(len(links)+1))
    return result


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    samples = corpus.copy()
    for i in samples:
        samples[i] = 0
    sample = None

    num_samples = n
    while num_samples > 0:
        if sample:
            distribution = transition_model(corpus, sample, damping_factor)
            # get lists of keys and values of the distribution dict
            pages = list(distribution.keys())
            weights = [distribution[i] for i in distribution]
            
            sample = random.choices(pages, weights, k = 1)[0]
        else:
            pages = list(corpus.keys())
            sample = random.choice(pages)
        samples[sample] += 1
        num_samples -= 1
    
    for i in samples:
        samples[i] /= n
    return samples



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    pages = {}
    for page in corpus:
        pages[page] = 1/num_pages
    new_pages = {}

    while True:

        for page in corpus:
            pr = 0
            # calculate sum
            for current_page in corpus:
                if len(corpus[current_page]) == 0:
                    pr += float(pages[current_page] / num_pages)
                if page in corpus[current_page]:
                    pr += float(pages[current_page] / len(corpus[current_page]))
            
            # calculate Pr(p) with p = page
            pr *= damping_factor
            pr += (1 - damping_factor) / num_pages
            new_pages[page] = pr
            
        diff = max([abs(pages[i] - new_pages[i]) for i in pages])
        if diff <= 0.001:
            break
        else:
            pages = new_pages.copy()

    return pages

    


if __name__ == "__main__":
    main()
