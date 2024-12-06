---
description: |
  Kleisli maps, and their weak equivalence with the Kleisli category.
---
<!--
```agda
open import Cat.Functor.FullSubcategory
open import Cat.Functor.Adjoint.Monad
open import Cat.Diagram.Monad.Solver
open import Cat.Functor.Conservative
open import Cat.Functor.Properties
open import Cat.Diagram.Terminal
open import Cat.Displayed.Total
open import Cat.Functor.Adjoint
open import Cat.Instances.Sets
open import Cat.Diagram.Monad
open import Cat.Prelude

open import Data.Bool

import Cat.Functor.Reasoning
import Cat.Reasoning
```
-->

```agda
module Cat.Diagram.Monad.Kleisli where
```

<!--
```agda
open Total-hom
```
-->

<!--
```agda
module _ {o ℓ} {C : Precategory o ℓ} {F : Functor C C} (M : Monad-on F) where
  private
    module M = Monad-on M
    module MR = Cat.Functor.Reasoning F
    module EM = Cat.Reasoning (Eilenberg-Moore M)
    module Free = Functor (Free-EM {M = M})

  open Cat.Reasoning C
  open M
```
-->

# Kleisli maps {defines="kleisli-morphism kleisli-map"}

Let $M : \cC \to \cC$ be a [[monad]]. A **Kleisli map**
from $X$ to $Y$ with respect to $M$ is a morphism $\cC(X, M(Y))$.
Like any reasonable notion of map, Kleisli maps can be organized into
a category.

```agda
  Kleisli-maps : Precategory o ℓ
  Kleisli-maps .Precategory.Ob = Ob
  Kleisli-maps .Precategory.Hom X Y = Hom X (M₀ Y)
  Kleisli-maps .Precategory.Hom-set _ _ = Hom-set _ _
```

The identity Kleisli map is simply the unit of the monad, and
composition of Kleisli maps is given by the following formula:
$$
  f \circ g := \mu \circ M(f) \circ g
$$

```agda
  Kleisli-maps .Precategory.id = η _
  Kleisli-maps .Precategory._∘_ f g = μ _ ∘ M₁ f ∘ g
```

<details>
<summary>The category equations all follow from simple yet tedious algebra.
Luckily, the algebra is simple enough that we can automate it away!
</summary>
```agda
  Kleisli-maps .Precategory.idr _ = lswizzle (sym (unit.is-natural _ _ _)) μ-idr
  Kleisli-maps .Precategory.idl _ = cancell μ-idl
  Kleisli-maps .Precategory.assoc _ _ _ = monad! M
```
</details>

Note that each Kleisli map $f : \cC(X,M(Y))$ can be extended
to an $M$-algebra homomorphism between [[free $M$-algebras|free-algebra]]
on $X$ and $Y$ via $\mu \circ M(f)$. This allows us to construct a functor
from the category of Kleisli maps into the [[Kleisli category]] of $M$.

```agda
  Kleisli-maps→Kleisli : Functor Kleisli-maps (Kleisli M)
  Kleisli-maps→Kleisli .Functor.F₀ X = Free.₀ X , inc (X , EM.id-iso)
  Kleisli-maps→Kleisli .Functor.F₁ f .hom = μ _ ∘ M₁ f
  Kleisli-maps→Kleisli .Functor.F₁ f .preserves = monad! M
```

<details>
<summary>We opt to omit the functoriality proofs, as they are not
particularly interesting.
</summary>

```agda
  Kleisli-maps→Kleisli .Functor.F-id =
    ext μ-idl
  Kleisli-maps→Kleisli .Functor.F-∘ f g =
    ext (MR.shufflel μ-assoc ∙ pushr (MR.shufflel (mult.is-natural _ _ _)))
```
</details>

A bit of quick algebra shows us that this functor is faithful.

```agda
  Kleisli-maps→Kleisli-is-faithful : is-faithful Kleisli-maps→Kleisli
  Kleisli-maps→Kleisli-is-faithful {_} {_} {f} {g} p =
    f                   ≡⟨ monad! M ⟩
    (μ _ ∘ M₁ f) ∘ η _  ≡⟨ ap₂ _∘_ (ap hom p) refl ⟩
    (μ _ ∘ M₁ g) ∘ η _  ≡⟨ monad! M ⟩
    g                   ∎
```

Moreover, this functor is full, as every $M$-algebra homomorphism
$f : M(X) \to M(Y)$ between free $M$-algebras can be transformed
into a Kleisli map by precomposing with the unit of the monad.

```agda
  Kleisli-maps→Kleisli-is-full : is-full Kleisli-maps→Kleisli
  Kleisli-maps→Kleisli-is-full {X} {Y} f =
    pure ((f .hom ∘ η _) , ext lemma)
    where
      lemma : μ Y ∘ M₁ (f .hom ∘ η X) ≡ f .hom
      lemma =
        μ _ ∘ M₁ (f .hom ∘ η _)   ≡⟨ MR.popl (sym (f .preserves)) ⟩
        (f .hom ∘ μ _) ∘ M₁ (η _) ≡⟨ cancelr μ-idl ⟩
        f .hom                    ∎
```

<!--
```agda
  Kleisli-maps→Kleisli-is-ff : is-fully-faithful Kleisli-maps→Kleisli
  Kleisli-maps→Kleisli-is-ff =
    full+faithful→ff Kleisli-maps→Kleisli
      Kleisli-maps→Kleisli-is-full
      Kleisli-maps→Kleisli-is-faithful
```
-->

Finally, this functor is [[essentially surjective]]. In other words,
the category of Kleisli maps is weakly equivalent to the Kleisli
category!

```agda
  Kleisli-maps→Kleisli-is-eso : is-eso Kleisli-maps→Kleisli
  Kleisli-maps→Kleisli-is-eso ((X , alg) , essentially-free) = do
    A , free ← essentially-free
    pure (A , (super-iso→sub-iso _ free))
```

Note that we **cannot** upgrade this weak equivalence to an equivalence when
$\cC$ is univalent, as categories of Kleisli maps are not, in general,
univalent. To see why, consider the monad on $\Sets$ that maps every
set to $\top$. Note that every hom set of the corresponding category of
Kleisli maps is contractible, as all maps are of the form $A \to \top$.
This in turn means that the type of isos is contractible; if the Kleisli
map category were univalent, then this would imply that the type of
sets was a proposition.

```agda
Kleisli-not-univalent
  : ∀ {κ}
  → Σ[ C ∈ Precategory (lsuc κ) κ ]
    Σ[ M ∈ Monad C ]
    (is-category C × (¬ is-category (Kleisli-maps (M .snd))))
Kleisli-not-univalent {κ} =
  Sets κ , (_ , T) , Sets-is-category , not-univalent
  where
    T : Monad-on _
    T =
      Adjunction→Monad $
      is-terminal→inclusion-is-right-adjoint (Sets κ)
        (el! (Lift _ ⊤))
        (λ _ → hlevel 0)

    open Cat.Reasoning (Kleisli-maps T)

    not-univalent : ¬ is-category (Kleisli-maps T)
    not-univalent cat =
      ¬Set-is-prop $
      identity-system→hlevel 0 cat $ λ X Y →
        ≅-is-contr (hlevel 0)
```

# Properties

<!--
```agda
module _ {o ℓ} {C : Precategory o ℓ} {F : Functor C C} {M : Monad-on F} where
  private
    module M = Monad-on M
    module MR = Cat.Functor.Reasoning F
    module EM = Cat.Reasoning (Eilenberg-Moore M)
    module Free = Functor (Free-EM {M = M})

  open Cat.Reasoning C
  open M
```
-->

As shown in the previous section, the category of Kleisli maps is weakly
equivalent to the Kleisli category, so it inherits all of the Kleisli
category's nice properties. In particular, the forgetful functor from
the category of Kleisli maps is faithful, conservative, and has a left
adjoint.

```agda
  Forget-Kleisli-maps : Functor (Kleisli-maps M) C
  Forget-Kleisli-maps = Forget-Kleisli F∘ Kleisli-maps→Kleisli M

  Forget-Kleisli-maps-is-faithful : is-faithful Forget-Kleisli-maps
  Forget-Kleisli-maps-is-faithful p =
    Kleisli-maps→Kleisli-is-faithful M (Forget-EM-is-faithful p)

  Forget-Kleisli-maps-is-conservative : is-conservative Forget-Kleisli-maps
  Forget-Kleisli-maps-is-conservative f-inv =
    is-ff→is-conservative {F = Kleisli-maps→Kleisli M} (Kleisli-maps→Kleisli-is-ff M) _ $
    super-inv→sub-inv _ $
    Forget-EM-is-conservative f-inv

  Free-Kleisli-maps : Functor C (Kleisli-maps M)
  Free-Kleisli-maps .Functor.F₀ X = X
  Free-Kleisli-maps .Functor.F₁ f = η _ ∘ f
  Free-Kleisli-maps .Functor.F-id = idr _
  Free-Kleisli-maps .Functor.F-∘ f g = monad! M

  Free-Kleisli-maps⊣Forget-Kleisli-maps : Free-Kleisli-maps ⊣ Forget-Kleisli-maps
  Free-Kleisli-maps⊣Forget-Kleisli-maps ._⊣_.unit ._=>_.η = η
  Free-Kleisli-maps⊣Forget-Kleisli-maps ._⊣_.unit ._=>_.is-natural _ _ f = monad! M
  Free-Kleisli-maps⊣Forget-Kleisli-maps ._⊣_.counit ._=>_.η _ = id
  Free-Kleisli-maps⊣Forget-Kleisli-maps ._⊣_.counit ._=>_.is-natural _ _ f = monad! M
  Free-Kleisli-maps⊣Forget-Kleisli-maps ._⊣_.zig = monad! M
  Free-Kleisli-maps⊣Forget-Kleisli-maps ._⊣_.zag = monad! M
```
