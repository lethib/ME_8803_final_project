# PE Consumption Website 🖥️

This is the folder that contains all the files to build the static github pages to publish my project online.

## Requirements ❗️

1. **Install Ruby**. You need to have Ruby installed to run the website locally on your machine. I have ruby 3.2.0.

2. **Install Jekyll**. Follow [these instructions](https://jekyllrb.com/docs/installation/) to install Jekyll based on your OS.

3. Once the repo cloned, install all the required gems:

   ```bash
   cd website
   bundle install
   ```

## Run a local server 👨🏽‍💻

To run a local server with jekyll, simply run:

```bash
bundle exec jekyll serve
```

## Project Structure 🗂️

- **\_data**: Contains the [meny.yml](./_data/menu.yml) file to have a custom navigation (as I have several pages for the EDA or the modeling parts)

- **\_includes**: Contains all the custom `.html` files used for the website template (head for SEO, footer, header and nav).

- **\_layouts**: Contains all the layouts for the website template.

- **\_sass**: Contains all the `.scss` files to style the website. These are all the original styles from the [minima]() template.

- **assets**: Contains all of the custom assets used for this website. It is structured like this:

  ```
  .
  └── assets/
    ├── css/
    |    └── style.scss
    └── img/
        └── favicon/
  ```

  The `style.scss` file is used to add custom styles to the websites. It will import the base styles of the template and add our custom scss styles.

  The [img/](./assets/img/) folder contains all the images used in the website (mainly plots). The subfolder [favicon/](./assets/img/favicon/) contains all the necessary files to have the website favicon across all devices and have also some SEO files.

- **pages**: Contains all the notebooks converted as mardown files. These are the base files that Jekyll will convert into html files for the website. It is structured like this:

  ```
  .
  └── pages/
    ├── EDA/
    ├── feature_engineering/
    └── modeling/
  ```

  Each subfolders ([EDA/](./pages/EDA/), [feature_engineering/](./pages/feature_engineering/) and [modeling/](./pages/modeling/)) contains at least an `index.md` file. This is required to have a custom navigation with the [menu.yml](./_data/menu.yml) file: each menu entries must be a folder within the [pages/](./pages/) folder with this file in it (it will be the landing page when you click on the nav link on the website).
