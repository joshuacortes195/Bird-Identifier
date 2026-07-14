# EDA — birds

- **Classes:** 555
- **Train images:** 21519
- **Images with bbox:** 48562
- **Imbalance ratio (train):** 18.0x
- **Image size (median W×H):** 1024×683

## Rarest classes

- Dark-eyed Junco (White-winged): 3
- Harlequin Duck (Female/juvenile): 4
- Boreal Chickadee: 5
- Barrow's Goldeneye (Female/Eclipse male): 6
- Bank Swallow: 7
- Greater Scaup (Female/Eclipse male): 8
- Fox Sparrow (Thick-billed/Slate-colored): 8
- Barrow's Goldeneye (Breeding male): 9

## Commonest classes

- House Sparrow (Female/Juvenile): 54
- American Goldfinch (Female/Nonbreeding Male): 54
- House Finch (Female/immature): 54
- Brewer's Blackbird (Male): 54
- Red-winged Blackbird (Female/juvenile): 54
- Northern Cardinal (Female/Juvenile): 54
- Magnolia Warbler (Female/immature male): 54
- American Redstart (Female/juvenile): 54

## Figures

![class distribution](class_distribution.png)
![image sizes](image_size_stats.png)
![samples](sample_grid.png)

> Hardest-confusable species pairs are surfaced quantitatively in Phase 6
> (confusion matrix), once a model exists.