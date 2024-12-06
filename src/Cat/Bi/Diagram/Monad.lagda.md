<!--
```agda
open import Cat.Bi.Base
open import Cat.Prelude

import Cat.Diagram.Monad as Cat
import Cat.Reasoning as Cr
```
-->

```agda
module Cat.Bi.Diagram.Monad  where
```

<!--
```agda
open _=>_

module _ {o ℓ ℓ'} (B : Prebicategory o ℓ ℓ') where
  private module B = Prebicategory B
```
-->

# Monads in a bicategory {defines="monad-in"}

Recall that a [monad] _on_ a category $\cC$ consists of a functor $M :
\cC \to \cC$ and natural transformations $\mu : MM \To M$, $\eta : \Id
\To M$. While the words "functor" and "natural transformation" are
specific to the setup where $\cC$ is a category, if we replace those
with "1-cell" and "2-cell", then the definition works in any
[[bicategory]]!

[monad]: Cat.Diagram.Monad.html

```agda
  record Monad (a : B.Ob) : Type (ℓ ⊔ ℓ') where
    field
      M : a B.↦ a
      μ : (M B.⊗ M) B.⇒ M
      η : B.id B.⇒ M
```

The setup is, in a sense, a lot more organic when phrased in an
arbitrary bicategory: Rather than dealing with the specificities of
natural transformations and the category $\cC$, we abstract all of
that away into the setup of the bicategory $\bf{B}$. All we need is that
the multiplication $\mu$ be compatible with the associator $\alpha$, and
the unit $\eta$ must be appropriately compatible with the left and right
unitors $\lambda, \rho$.

```agda
      μ-assoc : μ B.∘ (M B.▶ μ) ≡ μ B.∘ (μ B.◀ M) B.∘ B.α← M M M
      μ-unitr : μ B.∘ (M B.▶ η) ≡ B.ρ← M
      μ-unitl : μ B.∘ (η B.◀ M) ≡ B.λ← M
```

We can draw these compatibility conditions as pretty commutative
diagrams. The commutative altar (on top) indicates associativity of
multiplication, or more abstractly, compatibility of the multiplication
with the associator. The commutative upside-down triangle indicates
mutual compatibility of the multiplication and unit with the unitors.

<div class=mathpar>

~~~{.quiver}
\[\begin{tikzcd}
  & {(MM)M} && {M(MM)} \\
  \\
  MM && M && MM
  \arrow["\alpha", from=1-4, to=1-2]
  \arrow["{\mu\blacktriangleleft M}"', from=1-2, to=3-1]
  \arrow["{M \blacktriangleright \mu}"', from=1-4, to=3-5]
  \arrow["\mu", from=3-5, to=3-3]
  \arrow["\mu"', from=3-1, to=3-3]
\end{tikzcd}\]
~~~

~~~{.quiver}
\[\begin{tikzcd}
  {M\mathrm{Id}} && MM && {\mathrm{Id}M} \\
  \\
  && M
  \arrow["{M \blacktriangleright \eta}", from=1-1, to=1-3]
  \arrow["\mu", from=1-3, to=3-3]
  \arrow["\lambda"', from=1-1, to=3-3]
  \arrow["\rho", from=1-5, to=3-3]
  \arrow["{\eta \blacktriangleleft M}"', from=1-5, to=1-3]
\end{tikzcd}\]
~~~

</div>

## In Cat

To prove that this is an actual generalisation of the 1-categorical
notion, we push some symbols around and prove that a monad in the
bicategory $\bf{Cat}$ is the same thing as a monad _on_ some category.
Things are set up so that this is almost definitional, but the
compatibility paths have to be adjusted slightly. Check it out below:

```agda
module _ {o ℓ} {C : Precategory o ℓ} where
  open Cat.Monad-on
  open Monad
  private module C = Cr C

  Bicat-monad→monad : Monad (Cat _ _) C → Cat.Monad C
  Bicat-monad→monad monad = _ , monad' where
    private module M = Monad monad

    monad' : Cat.Monad-on M.M
    monad' .unit = M.η
    monad' .mult = M.μ
    monad' .μ-idl {x} =
        ap (M.μ .η x C.∘_) (C.intror refl)
      ∙ M.μ-unitr ηₚ x
    monad' .μ-idr {x} =
        ap (M.μ .η x C.∘_) (C.introl (M.M .Functor.F-id))
      ∙ M.μ-unitl ηₚ x
    monad' .μ-assoc {x} =
        ap (M.μ .η x C.∘_) (C.intror refl)
     ·· M.μ-assoc ηₚ x
     ·· ap (M.μ .η x C.∘_) (C.elimr refl ∙ C.eliml (M.M .Functor.F-id))

  Monad→bicat-monad : Cat.Monad C → Monad (Cat _ _) C
  Monad→bicat-monad (M , monad) = monad' where
    private module M = Cat.Monad-on (monad)

    monad' : Monad (Cat _ _) C
    monad' .Monad.M = M
    monad' .μ = M.mult
    monad' .η = M.unit
    monad' .μ-assoc = ext λ _ →
        ap (M.μ _ C.∘_) (C.elimr refl)
     ·· M.μ-assoc
     ·· ap (M.μ _ C.∘_) (C.introl (M.M-id) ∙ C.intror refl)
    monad' .μ-unitr = ext λ _ →
        ap (M.μ _ C.∘_) (C.elimr refl)
      ∙ M.μ-idl
    monad' .μ-unitl = ext λ _ →
        ap (M.μ _ C.∘_) (C.eliml M.M-id)
      ∙ M.μ-idr
```
