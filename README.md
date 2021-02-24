# Quantile-Quantile Embedding

## Paper

This is the code for the following paper:

- Benyamin Ghojogh, Fakhri Karray, and Mark Crowley. "**Quantile-Quantile Embedding for Distribution Transformation, Manifold Embedding, and Image Embedding with Choice of Embedding Distribution**" arXiv preprint arXiv:2006.11385 (2020).
  - Link of paper: [click here](https://arxiv.org/abs/2006.11385).

This code is for **Quantile-Quantile Embedding (QQE)**.

Some manifold learning and dimensionality reduction methods, such as PCA, Isomap, and MDS, do not care about the distribution of embedding. Some other manifold learning and dimensionality reduction methods, such as SNE and t-SNE, force the distribution of embedding to a specific distribution. They do not give choice of embedding distribution to the user. **QQE gives user the freedom to choose the distribution of embedding in manifold learning and dimensionality reduction. QQE also can be used for distribution transformation of data.**

## Using QQE for Distribution Transformation

### Synthetic Data with Reference Sample

![distribution_transform_synthetic2](https://user-images.githubusercontent.com/66282117/85348760-94bc8e00-b4ca-11ea-83a2-034a2a1bb3ed.png)

### Synthetic Data with Reference CDF

<img src="https://user-images.githubusercontent.com/66282117/85349033-5a072580-b4cb-11ea-8dc1-fb8bee15d5ab.png" width="70%">

### Facial Image Data (Changing Distribution to Have Eye-Glasses)

<img src="https://user-images.githubusercontent.com/66282117/85349091-83c04c80-b4cb-11ea-8c44-bcc87eb1f7fe.png" width="70%">

## Manifold Embedding

### Synthetic Data

![manifoldEmbedding_synthetic-1](https://user-images.githubusercontent.com/66282117/85349293-19f47280-b4cc-11ea-9feb-7f7e5ba4994e.png)

### MNIST Digit Data

![manifoldEmbedding_mnist-1](https://user-images.githubusercontent.com/66282117/85349138-abafb000-b4cb-11ea-8587-0945df97c696.png)

### An Example of Progress of Algorithm

![manifoldEmbedding_mnist_iterations-1](https://user-images.githubusercontent.com/66282117/85349171-bd915300-b4cb-11ea-8866-0f5a09195a83.png)

## Use of QQE for Separation of Classes

![class_separation-1](https://user-images.githubusercontent.com/66282117/85349190-d00b8c80-b4cb-11ea-8a99-a4219b9236f1.png)


