---
description: |
  We establish the crude monadicity theorem, which establishes
  sufficient conditions for an adjunction to be monadic.
---
<!--
```agda
open import Cat.Functor.Adjoint.Monadic
open import Cat.Functor.Adjoint.Monad
open import Cat.Functor.Conservative
open import Cat.Functor.Monadic.Beck
open import Cat.Diagram.Coequaliser
open import Cat.Functor.Equivalence
open import Cat.Displayed.Total
open import Cat.Functor.Adjoint
open import Cat.Diagram.Monad
open import Cat.Prelude

import Cat.Functor.Reasoning as F-r
import Cat.Reasoning as C-r
```
-->

```agda
module
  Cat.Functor.Monadic.Crude
  {o ℓ o' ℓ'} {C : Precategory o ℓ} {D : Precategory o' ℓ'}
  {F : Functor C D} {U : Functor D C}
  (F⊣U : F ⊣ U)
  where
```

<!--
```agda
private
  module F = F-r F
  module U = F-r U
  module C = C-r C
  module D = C-r D
  module UF = F-r (U F∘ F)
  module T = Monad-on (Adjunction→Monad F⊣U)

  T : Monad-on _
  T = Adjunction→Monad F⊣U
  C^T : Precategory _ _
  C^T = Eilenberg-Moore T

  module C^T = C-r C^T

open _⊣_ F⊣U
open _=>_
open Algebra-on
open Total-hom
```
-->

# Crude monadicity

We present a refinement of the conditions laid out in [Beck's
coequaliser] for when an adjunction $F \dashv G$ is [monadic]: The
**crude monadicity theorem**. The proof we present is adapted from
[@SIGL, chap. IV; sect. 4], where it is applied to the setting of
elementary topoi, but carried out in full generality.

[Beck's coequaliser]: Cat.Functor.Monadic.Beck.html#becks-coequaliser
[monadic]: Cat.Functor.Adjoint.Monadic.html

Recall our setup. We have a [[left adjoint]] functor $F : \cC \to
\cD$ (call its right adjoint $U$), and we're interested in
characterising exactly when the [comparison functor][comp] $K : \cD
\to \cC^{UF}$ into the [Eilenberg-Moore category][emc] of the [monad
from the adjunction][madj] is an equivalence. Our refinement here gives
a _sufficient_ condition.

[comp]: Cat.Functor.Adjoint.Monadic.html#Comparison-EM
[emc]: Cat.Diagram.Monad.html#eilenberg-moore-category
[madj]: Cat.Functor.Adjoint.Monad.html

**Theorem** (Crude monadicity). Let the functors $F$, $U$ be as in the
paragraph above, and abbreviate the resulting monad by $T$; Denote the
comparison functor by $K$.

1. If $\cD$ has [Beck's coequalisers] for any $T$-algebra, then $K$
has a left adjoint $K\inv \dashv K$;

2. If, in addition, $U$ preserves coequalisers for any pair which has a
common right inverse, then the unit of the adjunction $\eta$ is a
natural isomorphism;

3. If, in addition, $U$ is [conservative], then the counit of the
adjunction $\eta$ is also a natural isomorphism, and $K$ is an
[equivalence of precategories][eqv].

[Beck's coequalisers]: Cat.Functor.Monadic.Beck.html#presented-algebras
[conservative]: Cat.Functor.Conservative.html
[eqv]: Cat.Functor.Equivalence.html

**Proof** We already established (1) [here][Beck's coequalisers].

Let us show (2). Suppose that $\cD$ has Beck's coequalisers and that
$U$ preserves coequalisers of pairs of morphisms with a common right
inverse (we call these coequalisers **reflexive**):

```agda
private
  U-preserves-reflexive-coeqs : Type _
  U-preserves-reflexive-coeqs =
    ∀ {A B} {f g : D.Hom A B} (i : D.Hom B A)
    → (f D.∘ i ≡ D.id) → (g D.∘ i ≡ D.id)
    → (coequ : Coequaliser D f g)
    → is-coequaliser C (U.₁ f) (U.₁ g) (U.₁ (coequ .Coequaliser.coeq))

module _
  (has-coeq : (M : Algebra T) → Coequaliser D (F.₁ (M .snd .ν)) (counit.ε _))
  (U-pres : U-preserves-reflexive-coeqs)
  where
```

<!--
```agda
  open is-coequaliser
  open Coequaliser
  open Functor

  private
    K⁻¹ : Functor C^T D
    K⁻¹ = Comparison-EM⁻¹ F⊣U has-coeq

    K⁻¹⊣K : K⁻¹ ⊣ Comparison-EM F⊣U
    K⁻¹⊣K = Comparison-EM⁻¹⊣Comparison-EM F⊣U has-coeq

    module adj = _⊣_ K⁻¹⊣K
```
-->

```agda
  -- N.B.: In the code we abbreviate "preserves reflexive coequalisers"
  -- by "prcoeq".
  prcoeq→unit-is-iso : ∀ {o} → C^T.is-invertible (adj.unit.η o)
  prcoeq→unit-is-iso {o} = C^T.make-invertible inverse
    (ext η⁻¹η) (ext ηη⁻¹) where
```

The first thing we note is that Beck's coequaliser is reflexive: The
common right inverse is $F\eta$. It's straightforward to calculate that
this map _is_ a right inverse; let me point out that it follows from the
triangle identities for $F \dashv U$ and the algebra laws.

```agda
    abstract
      preserved : is-coequaliser C
        (UF.₁ (o .snd .ν)) (U.₁ (counit.ε (F.₀ (o .fst))))
        (U.₁ (has-coeq o .coeq))
      preserved =
        U-pres (F.₁ (unit.η _)) (F.annihilate (o .snd .ν-unit)) zig
          (has-coeq o)
```

It follows, since $U$ preserves coequalisers, that both rows of the diagram

~~~{.quiver}
\[\begin{tikzcd}
  {T^2o} & UFo & o \\
  {T^2o} & UFo & {UK\inv(o)}
  \arrow[shift left=1, from=1-1, to=1-2]
  \arrow[shift right=1, from=1-1, to=1-2]
  \arrow[shift left=1, from=2-1, to=2-2]
  \arrow[shift right=1, from=2-1, to=2-2]
  \arrow[Rightarrow, no head, from=1-1, to=2-1]
  \arrow[Rightarrow, from=1-2, to=2-2]
  \arrow["e", from=1-2, to=1-3]
  \arrow["Ue"', from=2-2, to=2-3]
  \arrow["{\eta_o\inv}", dashed, from=1-3, to=2-3]
\end{tikzcd}\]
~~~

are coequalisers, hence there is a unique isomorphism $\eta_o\inv$
making the diagram commute. This is precisely the inverse to $\eta_o$
we're seeking.

```agda
    η⁻¹ : C.Hom (U.₀ (coapex (has-coeq o))) (o .fst)
    η⁻¹η : adj.unit.η _ .hom C.∘ η⁻¹ ≡ C.id
    ηη⁻¹ : η⁻¹ C.∘ adj.unit.η _ .hom ≡ C.id

    η⁻¹ = preserved .universal {e' = o .snd .ν} (o .snd .ν-mult)

    η⁻¹η = is-coequaliser.unique₂ preserved
      {e' = U.₁ (has-coeq o .coeq)}
      (preserved .coequal)
      (C.pullr (preserved .factors)
       ∙ C.pullr (unit.is-natural _ _ _)
       ∙ C.pulll (preserved .coequal)
       ∙ C.cancelr zag)
      (C.idl _)

    ηη⁻¹ = C.pulll (preserved .factors) ∙ o .snd .ν-unit
```

It remains to show that $\eta\inv$ is a homomorphism of algebras. This
is a calculation reusing the established proof that $\eta\inv\eta =
\id$ established using the universal property of coequalisers above.

```agda
    inverse : C^T.Hom (U.₀ _ , _) o
    inverse .hom = η⁻¹
    inverse .preserves =
      η⁻¹ C.∘ U.₁ (counit.ε _)                                                              ≡⟨ C.refl⟩∘⟨ ap U.₁ (D.intror (F.annihilate (C.assoc _ _ _ ∙ η⁻¹η))) ⟩
      η⁻¹ C.∘ U.₁ (counit.ε _ D.∘ F.₁ (U.₁ (has-coeq o .coeq)) D.∘ F.₁ (unit.η _ C.∘ η⁻¹))  ≡⟨ C.refl⟩∘⟨ ap U.₁ (D.extendl (counit.is-natural _ _ _)) ⟩
      η⁻¹ C.∘ U.₁ (has-coeq o .coeq D.∘ counit.ε _ D.∘ F.₁ (unit.η _ C.∘ η⁻¹))              ≡⟨ C.refl⟩∘⟨ U.F-∘ _ _ ⟩
      η⁻¹ C.∘ U.₁ (has-coeq o .coeq) C.∘ U.₁ (counit.ε _ D.∘ F.₁ (unit.η _ C.∘ η⁻¹))        ≡⟨ C.pulll (preserved .factors) ⟩
      o .snd .ν C.∘ U.₁ (counit.ε _ D.∘ F.₁ (unit.η _ C.∘ η⁻¹))                             ≡⟨ C.refl⟩∘⟨ ap U.₁ (ap (counit.ε _ D.∘_) (F.F-∘ _ _) ∙ D.cancell zig) ⟩
      o .snd .ν C.∘ UF.₁ η⁻¹                                                                ∎
```

For (3), suppose additionally that $U$ is conservative. Recall that the
counit $\epsilon$ for the $K\inv \dashv K$ adjunction is defined as the
unique dotted map which fits into

~~~{.quiver}
\[\begin{tikzcd}
  FUFUA & FUA & {K\inv KA} \\
  && {A.}
  \arrow[two heads, from=1-2, to=1-3]
  \arrow["{\eps'_{FUA}}"', shift right=1, from=1-1, to=1-2]
  \arrow["{\eps'}"', from=1-2, to=2-3]
  \arrow["\epsilon", from=1-3, to=2-3]
  \arrow["{FU\eps_A'}", shift left=1, from=1-1, to=1-2]
\end{tikzcd}\]
~~~

But observe that the diagram

~~~{.quiver}
\[\begin{tikzcd}
  UFUFUA && UFUA & {UA,}
  \arrow[two heads, from=1-3, to=1-4]
  \arrow["{U\eps'_{FUA}}"', shift right=1, from=1-1, to=1-3]
  \arrow["{UFU\eps_A'}", shift left=1, from=1-1, to=1-3]
\end{tikzcd}\]
~~~

is _also_ a coequaliser; Hence, since $U$ preserves the coequaliser $FUA
\epi K\inv KA$, the map $U\eps : UK\inv KA \cong UA$; But by assumption
$U$ is conservative, so $\eps$ is an isomorphism, as desired.

```agda
  conservative-prcoeq→counit-is-iso
    : ∀ {o} → is-conservative U → D.is-invertible (adj.counit.ε o)
  conservative-prcoeq→counit-is-iso {o} reflect-iso = reflect-iso $
    C.make-invertible
      (U.₁ (coequ .coeq) C.∘ unit.η _) (U.pulll (coequ .factors) ∙ zag)
      inversel
    where

    oalg = Comparison-EM F⊣U .F₀ o
    coequ = has-coeq oalg

    abstract
      preserved : is-coequaliser C
        (UF.₁ (oalg .snd .ν)) (U.₁ (counit.ε (F.₀ (oalg .fst))))
        (U.₁ (coequ .coeq))
      preserved =
        U-pres (F.₁ (unit.η _)) (F.annihilate (oalg .snd .ν-unit)) zig coequ

    inversel =
      is-coequaliser.unique₂ preserved
        {e' = U.₁ (coequ .coeq)}
        (preserved .coequal)
        (C.pullr (U.collapse (coequ .factors))
            ∙ C.pullr (unit.is-natural _ _ _)
            ∙ C.pulll (preserved .coequal)
            ∙ C.cancelr zag)
        (C.idl _)
```

Packaging it all up, we get the claimed theorem: If $\cD$ has Beck's
coequalisers, and $U$ is a conservative functor which preserves
reflexive coequalisers, then the adjunction $F \dashv U$ is monadic.

```agda
crude-monadicity
  : (has-coeq : (M : Algebra T) → Coequaliser D (F.₁ (M .snd .ν)) (counit.ε _))
    (U-pres : U-preserves-reflexive-coeqs)
    (U-conservative : is-conservative U)
  → is-monadic F⊣U
crude-monadicity coeq pres cons = eqv' where
  open is-equivalence
  eqv : is-equivalence (Comparison-EM⁻¹ F⊣U coeq)
  eqv .F⁻¹          = Comparison-EM F⊣U
  eqv .F⊣F⁻¹        = Comparison-EM⁻¹⊣Comparison-EM F⊣U coeq
  eqv .unit-iso _   = prcoeq→unit-is-iso coeq pres
  eqv .counit-iso _ = conservative-prcoeq→counit-is-iso coeq pres cons
  eqv' = inverse-equivalence eqv
```
