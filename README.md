# traffic-sign-recognition-using-CNN

### :mortar_board: Seminarski rad u okviru kursa Računarska inteligencija na 4. godini I smera.


U ovom radu, prikazaćemo implementaciju za prepoznavanje saobraćajnih znakova koristeći CNN - konvolutivne neuronske mreže. Takođe, biće prikazano nekoliko CNN arhitektura, koje ćemo međusobno upoređivati. Treniranje neuronske mreže je implementirano pomoću Keras biblioteke.


Testiranje se pokreće komandom **python3 test.py** nakon čega je moguće navesti jednu od dve opcije:

| Opcija          | Značenje opcije                              |
| --------------- | ---------------------------------------------|
| -all            | testira sve slike iz test skupa              |
| -one            | testira jednu sliku čiji je naziv prosleđen  |

Treniranje se pokreće komandom **python3 train.py** nakon čega je moguće navesti jednu od tri opcije:

| Opcija          | Značenje opcije                                |
| --------------- | -----------------------------------------------|
| -M              | trenira model pomoću mreže koju smo mi smislie |
| -L              | trenira model pomoću LeNet mreže               |
| -A              | trenira model pomoću AlexNet mreže             |


### :eyes: Autori:

Jana Jovičić | Jovana Pejkić
