<!--
```agda
open import Cat.Displayed.Cocartesian
open import Cat.Displayed.Fibre
open import Cat.Displayed.Base
open import Cat.Prelude

import Cat.Displayed.Reasoning
```
-->

```agda
module Cat.Displayed.Cocartesian.Indexing
  {o ℓ o' ℓ'} {ℬ : Precategory o ℓ}
  (ℰ : Displayed ℬ o' ℓ')
  (opfibration : Cocartesian-fibration ℰ)
  where
```

<!--
```agda
open Precategory ℬ
open Displayed ℰ
open Cat.Displayed.Reasoning ℰ
open Cocartesian-fibration opfibration
open Functor
```
-->

# Opreindexing for cocartesian fibrations {defines="cobase-change"}

[Opfibrations] are dual to [fibrations], so they inherit the ability
to [reindex along morphisms in the base]. However, this reindexing is
*covariant* for opfibrations, whereas it is *contravariant* for
fibrations.

[Opfibrations]: Cat.Displayed.Cocartesian.html
[fibrations]: Cat.Displayed.Cartesian.html
[reindex along morphisms in the base]: Cat.Displayed.Cartesian.Indexing.html

This difference in variance gives opfibrations a distinct character.
Reindexing in fibrations can be thought of a sort of restriction map.
This can be seen clearly with [[canonical self-indexing]], where the
reindexing functors are given by [[pullbacks]]. On the other hand,
opreindexing can be thought of as an extension map. We can again use the
the canonical self-indexing as our example: opreindexing is given by
postcomposition, which extends families over $X$ to families over $Y$ by
adding in empty fibres.

[pullbacks]: Cat.Diagram.Pullback.html

```agda
cobase-change : ∀ {x y} (f : Hom x y) → Functor (Fibre ℰ x) (Fibre ℰ y)
cobase-change f .F₀ ob = has-lift.y' f ob
cobase-change f .F₁ vert = rebase f vert
```

<!--
```agda
cobase-change f .F-id =
  sym $ has-lift.uniquev _ _ _ $ to-pathp $
    idl[] ∙ (sym $ cancel _ _ (idr' _))
cobase-change f .F-∘ f' g' =
  sym $ has-lift.uniquev _ _ _ $ to-pathp $
    smashl _ _
    ·· revive₁ (pullr[] _ (has-lift.commutesv _ _ _))
    ·· smashr _ _
    ·· revive₁ (pulll[] _ (has-lift.commutesv _ _ _))
      ·· smashl _ _
      ·· sym assoc[]
      ·· sym (smashr _ _)
```
-->
