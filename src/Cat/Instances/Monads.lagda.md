<!--
```agda
open import Cat.Instances.Product
open import Cat.Displayed.Total
open import Cat.Functor.Compose
open import Cat.Displayed.Base
open import Cat.Diagram.Monad
open import Cat.Functor.Base
open import Cat.Prelude

import Cat.Reasoning

open Precategory
open Displayed
open Functor
```
-->

```agda
module Cat.Instances.Monads where
```

<!--
```agda
private variable
  o h : Level
```
-->

# Categories of monads {defines="category-of-monads"}

The **category of monads** of a category $\cC$ consists of [[monads]]
on $\cC$ and natural transformations preserving the monadic structure.
In terms of the bicategory of [[monads in $\Cat$|monad in]], a morphism
in the 1-categorical version always has the identity functor as the
underlying 1-cell. That is, this 1-category of monads on $\cC$ is the
fibre of the displayed bicategory of monads in $\Cat$ over $\cC$.

<!--
```agda
module _ {C : Precategory o h} where
  private
    module C = Cat.Reasoning C

    variable
      F G H : Functor C C
      M N O : Monad-on F

    Endo : Precategory (o ⊔ h) (o ⊔ h)
    Endo = Cat[ C , C ]
    module Endo = Cat.Reasoning Endo

    Endo-∘-functor : Functor (Endo ×ᶜ Endo) Endo
    Endo-∘-functor = F∘-functor
    module Endo-∘ = Functor Endo-∘-functor
```
-->

A monad homomorphism is a natural transformation $\nu$ preserving
the unit $\eta$ and the multiplication $\mu$. In other words, the
following two diagrams commute, where $\blacklozenge$ is the
[[horizontal composition|horizontal composition in cat]]:

~~~{.quiver}
\[\begin{tikzcd}
  {\Id} && {\Id} && {MM} && {NN} \\
  \\
  {M} && {N} && {M} && {N}
  \arrow["{\id}", from=1-1, to=1-3]
  \arrow["{\nu}"', from=3-1, to=3-3]
  \arrow["{\eta_M}"', from=1-1, to=3-1]
  \arrow["{\eta_N}", from=1-3, to=3-3]
  \arrow["{\nu\mathbin{\blacklozenge}\nu}", from=1-5, to=1-7]
  \arrow["{\nu}"', from=3-5, to=3-7]
  \arrow["{\mu_M}"', from=1-5, to=3-5]
  \arrow["{\mu_N}", from=1-7, to=3-7]
\end{tikzcd}\]
~~~

```agda
  record is-monad-hom (nat : F => G) (M : Monad-on F) (N : Monad-on G) : Type (o ⊔ h) where
    no-eta-equality
```

<!--
```agda
    private
      module M = Monad-on M
      module N = Monad-on N
    open _=>_ nat public
```
-->

```agda
    field
      pres-unit : nat ∘nt M.unit ≡ N.unit
      pres-mult : nat ∘nt M.mult ≡ N.mult ∘nt (nat ◆ nat)
```

<!--
```agda
  abstract instance
    H-Level-is-monad-hom : ∀ {eta n} → H-Level (is-monad-hom eta M N) (suc n)
    H-Level-is-monad-hom = prop-instance $ Iso→is-hlevel 1 eqv (hlevel 1)
      where unquoteDecl eqv = declare-record-iso eqv (quote is-monad-hom)

  open is-monad-hom using (pres-unit ; pres-mult)
```
-->

These contain the identity and are closed under composition, as expected.

```agda
  id-is-monad-hom : {F : Functor C C} {M : Monad-on F} → is-monad-hom idnt M M
  id-is-monad-hom {M = _} .pres-unit = Endo.idl _
  id-is-monad-hom {M = M} .pres-mult =
    let module M = Monad-on M in
    idnt ∘nt M.mult          ≡⟨ Endo.id-comm-sym ⟩
    M.mult ∘nt idnt          ≡˘⟨ ap (M.mult ∘nt_) Endo-∘.F-id ⟩
    M.mult ∘nt (idnt ◆ idnt) ∎

  ∘-is-monad-hom
    : ∀ {ν : G => H} {ξ : F => G}
    → is-monad-hom ν N O → is-monad-hom ξ M N → is-monad-hom (ν ∘nt ξ) M O
  ∘-is-monad-hom {N = N} {O} {M} {ν} {ξ} p q = mh where
    module M = Monad-on M
    module N = Monad-on N
    module O = Monad-on O
    module ν = is-monad-hom p
    module ξ = is-monad-hom q

    mh : is-monad-hom _ M O
    mh .pres-unit =
      (ν ∘nt ξ) ∘nt M.unit ≡˘⟨ Endo.assoc ν ξ M.unit ⟩
      ν ∘nt (ξ ∘nt M.unit) ≡⟨ ap (ν ∘nt_) ξ.pres-unit ⟩
      ν ∘nt N.unit         ≡⟨ ν.pres-unit ⟩
      O.unit               ∎
    mh .pres-mult =
      (ν ∘nt ξ) ∘nt M.mult               ≡˘⟨ Endo.assoc ν ξ M.mult ⟩
      ν ∘nt (ξ ∘nt M.mult)               ≡⟨ ap (ν ∘nt_) ξ.pres-mult ⟩
      ν ∘nt (N.mult ∘nt (ξ ◆ ξ))         ≡⟨ Endo.assoc ν N.mult (ξ ◆ ξ) ⟩
      (ν ∘nt N.mult) ∘nt (ξ ◆ ξ)         ≡⟨ ap (_∘nt (ξ ◆ ξ)) ν.pres-mult ⟩
      (O.mult ∘nt (ν ◆ ν)) ∘nt (ξ ◆ ξ)   ≡˘⟨ Endo.assoc O.mult (ν ◆ ν) (ξ ◆ ξ) ⟩
      O.mult ∘nt ((ν ◆ ν) ∘nt (ξ ◆ ξ))   ≡˘⟨ ap (O.mult ∘nt_) $ Endo-∘.F-∘ (ν , ν) (ξ , ξ) ⟩
      O.mult ∘nt ((ν ∘nt ξ) ◆ (ν ∘nt ξ)) ∎
```

Putting these together, we have the category of monads.

```agda
Monads' : (C : Precategory o h) → Displayed Cat[ C , C ] _ _
Monads' C .Ob[_] = Monad-on
Monads' C .Hom[_] = is-monad-hom
Monads' C .Hom[_]-set _ _ _ = hlevel 2
Monads' C .id' = id-is-monad-hom
Monads' C ._∘'_ = ∘-is-monad-hom
Monads' C .idr' _ = prop!
Monads' C .idl' _ = prop!
Monads' C .assoc' _ _ _ = prop!
```

<!--
```agda
Monads : Precategory o h → Precategory _ _
Monads C = ∫ (Monads' C)
```
-->
