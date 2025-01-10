---
description: |
  Comma categories as two-sided displayed categories.
---
<!--
```agda
open import Cat.Displayed.TwoSided.Discrete
open import Cat.Displayed.Cocartesian
open import Cat.Displayed.Cartesian
open import Cat.Instances.Product
open import Cat.Displayed.Base
open import Cat.Prelude

import Cat.Functor.Reasoning
import Cat.Reasoning
```
-->
```agda
module Cat.Displayed.Instances.Comma where
```

# Comma categories as displayed categories

We can neatly present [[comma categories]] as categories displayed over
product categories.

<!--
```agda
module _
  {oa ℓa ob ℓb oc ℓc}
  {A : Precategory oa ℓa}
  {B : Precategory ob ℓb}
  {C : Precategory oc ℓc}
  (F : Functor A C)
  (G : Functor B C)
  where
  private
    module A = Cat.Reasoning A
    module B = Cat.Reasoning B
    module C = Cat.Reasoning C
    module F = Cat.Functor.Reasoning F
    module G = Cat.Functor.Reasoning G

  open Displayed
```
-->

```agda
  Comma : Displayed (A ×ᶜ B) ℓc ℓc
  Comma .Ob[_] (a , b) = C.Hom (F.₀ a) (G.₀ b)
  Comma .Hom[_] (u , v) f g = G.₁ v C.∘ f ≡ g C.∘ F.₁ u
  Comma .Hom[_]-set _ _ _ = hlevel 2
  Comma .id' = C.eliml G.F-id ∙ C.intror F.F-id
  Comma ._∘'_ α β = G.popr β ∙ sym (F.shufflel (sym α))
  Comma .idr' _ = prop!
  Comma .idl' _ = prop!
  Comma .assoc' _ _ _ = prop!
```

## Comma categories are discrete two-sided fibrations

<!--
```agda
module _
  {oa ℓa ob ℓb oc ℓc}
  {A : Precategory oa ℓa}
  {B : Precategory ob ℓb}
  {C : Precategory oc ℓc}
  {F : Functor A C}
  {G : Functor B C}
  where
  private
    module A = Cat.Reasoning A
    module B = Cat.Reasoning B
    module C = Cat.Reasoning C
    module F = Cat.Functor.Reasoning F
    module G = Cat.Functor.Reasoning G

  open is-discrete-two-sided-fibration
  open Displayed
```
-->

Comma categories are [[discrete two-sided fibrations]].

```agda
  Comma-is-discrete-two-sided-fibration
    : is-discrete-two-sided-fibration (Comma F G)
  Comma-is-discrete-two-sided-fibration .fibre-set _ _ = hlevel 2
  Comma-is-discrete-two-sided-fibration .cart-lift f g .centre =
    g C.∘ F.₁ f , C.eliml G.F-id
  Comma-is-discrete-two-sided-fibration .cart-lift f g .paths (h , p) =
    Σ-prop-path! (sym p ∙ C.eliml G.F-id)
  Comma-is-discrete-two-sided-fibration .cocart-lift f g .centre =
    G.₁ f C.∘ g , C.intror F.F-id
  Comma-is-discrete-two-sided-fibration .cocart-lift f g .paths (h , p) =
    Σ-prop-path! (p ∙ C.elimr F.F-id)
  Comma-is-discrete-two-sided-fibration .vert-lift {x = f} {y = g} {u = h} {v = k} p =
    G.F₁ B.id C.∘ G.F₁ k C.∘ f   ≡⟨ C.eliml G.F-id ⟩
    G.F₁ k C.∘ f                 ≡⟨ p ⟩
    g C.∘ F.₁ h                  ≡⟨ C.intror F.F-id ⟩
    (g C.∘ F.F₁ h) C.∘ F.F₁ A.id ∎
  Comma-is-discrete-two-sided-fibration .factors _ = prop!
```

Every $B$-vertical morphism in a discrete two-sided fibration is [[cartesian|cartesian-morphism]],
but this is not neccesarily true of *every* morphism. For instance, cartesian
maps in comma categories are precisely squares that satisfy the following
pasting property: for every (potentially non-commuting) square of the below
form, if the outer square commutes, then the upper square commutes.

~~~{.quiver}
\begin{tikzcd}
  F(A) && G(Y) \\
  \\
  F(W) && G(Y) \\
  \\
  F(X) && G(Z)
  \arrow["h", from=1-1, to=1-3]
  \arrow["F(j)"', from=1-1, to=3-1]
  \arrow["G(k)", from=1-3, to=3-3]
  \arrow["f", from=3-1, to=3-3]
  \arrow["F(u)"', from=3-1, to=5-1]
  \arrow["G(v)", from=3-3, to=5-3]
  \arrow["g"', from=5-1, to=5-3]
\end{tikzcd}
~~~

```agda
  pasting→comma-cartesian
    : ∀ {w x y z} {u : A.Hom w x} {v : B.Hom y z} {f : C.Hom (F.₀ w) (G.₀ y)} {g : C.Hom (F.₀ x) (G.₀ z)}
    → (p : G.₁ v C.∘ f ≡ g C.∘ F.₁ u)
    → (∀ {a b} {h : C.Hom (F.₀ a) (G.₀ b)} {j : A.Hom a w} {k : B.Hom b y}
       → G.₁ v C.∘ G.₁ k C.∘ h ≡ g C.∘ F.₁ u C.∘ F.₁ j
       → G.₁ k C.∘ h ≡ f C.∘ F.₁ j)
    → is-cartesian (Comma F G) (u , v) p
  pasting→comma-cartesian p paste .is-cartesian.universal _ outer =
    paste (C.pulll (sym $ G.F-∘ _ _) ·· outer ·· ap₂ C._∘_ refl (F.F-∘ _ _))
  pasting→comma-cartesian p paste .is-cartesian.commutes _ _ = prop!
  pasting→comma-cartesian p paste .is-cartesian.unique _ _ = prop!

  comma-cartesian→pasting
    : ∀ {w x y z} {u : A.Hom w x} {v : B.Hom y z} {f : C.Hom (F.₀ w) (G.₀ y)} {g : C.Hom (F.₀ x) (G.₀ z)}
    → (p : G.₁ v C.∘ f ≡ g C.∘ F.₁ u)
    → is-cartesian (Comma F G) (u , v) p
    → ∀ {a b} {h : C.Hom (F.₀ a) (G.₀ b)} {j : A.Hom a w} {k : B.Hom b y}
       → G.₁ v C.∘ G.₁ k C.∘ h ≡ g C.∘ F.₁ u C.∘ F.₁ j
       → G.₁ k C.∘ h ≡ f C.∘ F.₁ j
  comma-cartesian→pasting p p-cart outer =
    is-cartesian.universal p-cart _ $
      C.pushl (G.F-∘ _ _) ·· outer ·· ap₂ C._∘_ refl (sym $ F.F-∘ _ _)
```

Moreover, a square is [[cocartesian|cocartesian-morphism]] when it satisfies
the dual pasting lemma.

```agda
  pasting→comma-cocartesian
    : ∀ {w x y z} {u : A.Hom w x} {v : B.Hom y z} {f : C.Hom (F.₀ w) (G.₀ y)} {g : C.Hom (F.₀ x) (G.₀ z)}
    → (p : G.₁ v C.∘ f ≡ g C.∘ F.₁ u)
    → (∀ {a b} {h : C.Hom (F.₀ a) (G.₀ b)} {j : A.Hom x a} {k : B.Hom z b}
        → G.₁ k C.∘ G.₁ v C.∘ f ≡ h C.∘ F.₁ j C.∘ F.₁ u
        → G.₁ k C.∘ g ≡ h C.∘ F.₁ j)
    → is-cocartesian (Comma F G) (u , v) p

  comma-cocartesian→pasting
    : ∀ {w x y z} {u : A.Hom w x} {v : B.Hom y z} {f : C.Hom (F.₀ w) (G.₀ y)} {g : C.Hom (F.₀ x) (G.₀ z)}
    → (p : G.₁ v C.∘ f ≡ g C.∘ F.₁ u)
    → is-cocartesian (Comma F G) (u , v) p
    → ∀ {a b} {h : C.Hom (F.₀ a) (G.₀ b)} {j : A.Hom x a} {k : B.Hom z b}
        → G.₁ k C.∘ G.₁ v C.∘ f ≡ h C.∘ F.₁ j C.∘ F.₁ u
        → G.₁ k C.∘ g ≡ h C.∘ F.₁ j

```

<details>
<summary>The proofs are formally dual, so we omit them.
</summary>

```agda
  pasting→comma-cocartesian p paste .is-cocartesian.universal _ outer =
    paste (C.pulll (sym $ G.F-∘ _ _) ·· outer ·· ap₂ C._∘_ refl (F.F-∘ _ _))
  pasting→comma-cocartesian p paste .is-cocartesian.commutes _ _ = prop!
  pasting→comma-cocartesian p paste .is-cocartesian.unique _ _ = prop!

  comma-cocartesian→pasting p p-cocart outer =
    is-cocartesian.universal p-cocart _ $
      C.pushl (G.F-∘ _ _) ·· outer ·· ap₂ C._∘_ refl (sym $ F.F-∘ _ _)
```
</details>
