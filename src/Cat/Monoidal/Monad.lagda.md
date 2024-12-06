<!--
```agda
open import Cat.Monoidal.Strength.Monad
open import Cat.Functor.Naturality
open import Cat.Instances.Product
open import Cat.Monoidal.Strength
open import Cat.Monoidal.Braided
open import Cat.Monoidal.Reverse
open import Cat.Functor.Compose
open import Cat.Diagram.Monad
open import Cat.Monoidal.Base
open import Cat.Functor.Base
open import Cat.Prelude

import Cat.Functor.Reasoning
import Cat.Monoidal.Functor
import Cat.Reasoning
```
-->

```agda
module Cat.Monoidal.Monad where
```

# Monoidal monads {defines="monoidal-monad"}

<!--
```agda
module _ {o ℓ}
  {C : Precategory o ℓ} (Cᵐ : Monoidal-category C)
  where
  open Cat.Reasoning C
  open Cat.Monoidal.Functor Cᵐ Cᵐ
  open Monoidal-category Cᵐ
```
-->

A **monoidal monad** on a [[monoidal category]] $\cC$ is a [[monad]] on
$\cC$ whose underlying endofunctor is also equipped with the structure
of a [[lax monoidal functor]], in a way that the two structures agree;
alternatively, it is a [[monad *in*]] the [[bicategory]] of
monoidal categories, lax monoidal functors and [[monoidal natural
transformations]].

```agda
  record Monoidal-monad-on {F : Functor C C} (monad : Monad-on F) : Type (o ⊔ ℓ) where
    open Monad-on monad
    field
      monad-monoidal : Lax-monoidal-functor-on F

    open Lax-monoidal-functor-on monad-monoidal renaming
      (F-mult to M-mult; F-α← to M-α←; F-α→ to M-α→; F-λ← to M-λ←; F-ρ← to M-ρ←)
      public
```

To summarise, we have the following data laying around:

$$
\begin{align*}
\eta_A &: A \to M A & \mu_A &: M^2 A \to M A \\
\epsilon &: 1 \to M 1 & \varphi_{A,B} &: M A \otimes M B \to M (A \otimes B)
\end{align*}
$$

The precise interaction we ask for is that the monad's unit and
multiplication be [[monoidal natural transformations]]^[With respect to
the induced lax monoidal structures on $\Id$ and $M^2$.]; this requires a
total of *four* conditions.

That $\eta$ is a monoidal natural transformation is captured by the
following two diagrams:

<div class="mathpar">

~~~{.quiver}
\[\begin{tikzcd}[column sep=tiny]
  & 1 \\
  1 && M1
  \arrow[Rightarrow, no head, from=1-2, to=2-1]
  \arrow["\epsilon", from=1-2, to=2-3]
  \arrow["{\eta_1}"', from=2-1, to=2-3]
\end{tikzcd}\]
~~~

~~~{.quiver}
\[\begin{tikzcd}
  {A \otimes B} & {MA \otimes MB} \\
  {A \otimes B} & {M (A \otimes B)}
  \arrow[Rightarrow, no head, from=1-1, to=2-1]
  \arrow["{\eta_A \otimes \eta_B}", from=1-1, to=1-2]
  \arrow["{\varphi_{A,B}}", from=1-2, to=2-2]
  \arrow["{\eta_{A \otimes B}}"', from=2-1, to=2-2]
\end{tikzcd}\]
~~~

</div>

```agda
    field
      unit-ε : ε ≡ η Unit
      unit-φ : ∀ {A B} → φ {A} {B} ∘ (η A ⊗₁ η B) ≡ η (A ⊗ B)
```

Notice that the diagram on the left entirely determines $\epsilon$ to be
$\eta_1$.

Then, for $\mu$ to be monoidal means that these two diagrams commute:

<div class="mathpar">

~~~{.quiver}
\[\begin{tikzcd}[column sep=tiny]
  & 1 \\
  {M^21} && M1
  \arrow["{M\epsilon \circ \epsilon}"', from=1-2, to=2-1]
  \arrow["{\mu_1}"', from=2-1, to=2-3]
  \arrow["\epsilon", from=1-2, to=2-3]
\end{tikzcd}\]
~~~

~~~{.quiver}
\[\begin{tikzcd}
  {M^2 A \otimes M^2 B} & {MA \otimes MB} \\
  {M (MA \otimes MB)} \\
  {M^2 (A \otimes B)} & {M (A \otimes B)}
  \arrow["{\varphi_{MA,MB}}"', from=1-1, to=2-1]
  \arrow["{M\varphi_{A,B}}"', from=2-1, to=3-1]
  \arrow["{\mu_A \otimes \mu_B}", from=1-1, to=1-2]
  \arrow["{\varphi_{A,B}}", from=1-2, to=3-2]
  \arrow["{\mu_{A \otimes B}}"', from=3-1, to=3-2]
\end{tikzcd}\]
~~~

</div>

The diagram on the left automatically commutes because of the monad laws,
since $\epsilon$ is forced to be $\eta_1$, so we only need to require
the diagram on the right. This is the biggest requirement: so far, we're
describing a structure that any [[strong monad]] could be equipped with,
but making the last diagram commute requires a [[commutative monad]],
as we will see in the rest of this module.

```agda
    mult-ε : ε ≡ μ Unit ∘ M₁ ε ∘ ε
    mult-ε = insertl (ap (λ x → _ ∘ M₁ x) unit-ε ∙ μ-idl)

    field
      mult-φ : ∀ {A B} → φ {A} {B} ∘ (μ A ⊗₁ μ B) ≡ μ (A ⊗ B) ∘ M₁ φ ∘ φ

  Monoidal-monad : Type (o ⊔ ℓ)
  Monoidal-monad = Σ[ M ∈ Functor C C ] Σ[ M' ∈ Monad-on M ] (Monoidal-monad-on M')
```

<!--
```agda
  private unquoteDecl eqv = declare-record-iso eqv (quote Monoidal-monad-on)
  Monoidal-monad-on-path
    : ∀ {F} {M : Monad-on F} {a b : Monoidal-monad-on M}
    → a .Monoidal-monad-on.monad-monoidal ≡ b .Monoidal-monad-on.monad-monoidal
    → a ≡ b
  Monoidal-monad-on-path p = Iso.injective eqv (Σ-prop-path (λ _ → hlevel 1) p)
```
-->

## Duality

<!--
```agda
module _ {o ℓ}
  {C : Precategory o ℓ} {Cᵐ : Monoidal-category C} {M : Functor C C}
  {monad : Monad-on M}
  where
  open Cat.Reasoning C
  open Cat.Monoidal.Functor Cᵐ Cᵐ
  open Monoidal-category Cᵐ
  open Monad-on monad
  private
    module M = Cat.Functor.Reasoning M
```
-->

As the definition of a monoidal monad is completely symmetric with
respect to the tensor product, we straightforwardly get a monoidal
monad on the [[reverse monoidal category]] $\cC^\rm{rev}$ from a
monoidal monad on $\cC$.

```agda
  monoidal-monad^rev : Monoidal-monad-on Cᵐ monad → Monoidal-monad-on (Cᵐ ^rev) monad
  monoidal-monad^rev m = record
    { monad-monoidal = record
      { ε = ε
      ; F-mult = NT (λ _ → φ) λ _ _ _ → M-mult ._=>_.is-natural _ _ _
      ; F-α→ = M-α←
      ; F-λ← = M-ρ←
      ; F-ρ← = M-λ←
      }
    ; unit-ε = unit-ε
    ; unit-φ = unit-φ
    ; mult-φ = mult-φ
    } where open Monoidal-monad-on m
```

## As commutative monads

We now turn to proving the following result, alluded to above: the
structure of a monoidal monad on a given monad is *equivalent* to the
structure of a [[commutative strength]] for that monad!

::: source
This is a slight generalisation of a result by Kock [-@Kock:monoidal-monads],
who proves that *symmetric* monoidal monads on a [[symmetric monoidal
category]] correspond to commutative strengths (which are assumed to be
symmetric).
:::

The guiding intuition is that the monoidal structure on $M$ allows
*parallel*, or "vertical" composition of effects; this is illustrated by
the `mult-φ`{.Agda} diagram if we label the monadic "layers" as follows:

~~~{.quiver}
\[\begin{tikzcd}
  {M_1 M_2 A \otimes M_3 M_4 B} & {M_{12} A \otimes M_{34} B} \\
  {M^1_3 (M_2 A \otimes M_4 B)} \\
  {M^1_3 M^2_4 (A \otimes B)} & {M^{12}_{34} (A \otimes B)}
  \arrow["{\varphi_{MA,MB}}"', from=1-1, to=2-1]
  \arrow["{M\varphi_{A,B}}"', from=2-1, to=3-1]
  \arrow["{\mu_A \otimes \mu_B}", from=1-1, to=1-2]
  \arrow["{\varphi_{A,B}}", from=1-2, to=3-2]
  \arrow["{\mu_{A \otimes B}}"', from=3-1, to=3-2]
\end{tikzcd}\]
~~~

This should be reminiscent of the two composition structures in the
hypotheses to the [[Eckmann-Hilton argument]], which suggests that the
two ways to sequence effects ("monadically" and "monoidally") should be
the same, and, moreover, commutative.

### Commutative monads from monoidal monads

We start by constructing a [[left monad strength]] from a monoidal monad,
by composing the unit of the monad with the monoidal multiplication:

~~~{.quiver}
\[\begin{tikzcd}
  {A \otimes MB} & {MA \otimes MB} & {M (A \otimes B)}
  \arrow["{\eta_A \otimes MB}", from=1-1, to=1-2]
  \arrow["{\varphi_{A,B}}", from=1-2, to=1-3]
\end{tikzcd}\]
~~~

```agda
  monoidal→left-strength
    : Monoidal-monad-on Cᵐ monad → Left-monad-strength Cᵐ monad
  monoidal→left-strength m = l where
    open Monoidal-monad-on m
    open Left-strength
    open Left-monad-strength

    l : Left-monad-strength Cᵐ monad
    l .functor-strength .left-strength = M-mult ∘nt (-⊗- ▸ (unit nt× idnt))
```

<details>
<summary>
We leave the verification of the strength axioms for the curious
reader; they are entirely monotonous.
</summary>

```agda
    l .functor-strength .left-strength-λ← =
      M₁ λ← ∘ φ ∘ (η _ ⊗₁ id) ≡˘⟨ refl⟩∘⟨ refl⟩∘⟨ ap (_⊗₁ id) unit-ε ⟩
      M₁ λ← ∘ φ ∘ (ε ⊗₁ id)   ≡⟨ M-λ← ⟩
      λ←                      ∎
    l .functor-strength .left-strength-α→ =
      M₁ (α→ _ _ _) ∘ φ ∘ (η _ ⊗₁ id)                          ≡˘⟨ refl⟩∘⟨ refl⟩∘⟨ ◀.collapse unit-φ ⟩
      M₁ (α→ _ _ _) ∘ φ ∘ (φ ⊗₁ id) ∘ ((η _ ⊗₁ η _) ⊗₁ id)     ≡⟨ extendl3 M-α→ ⟩
      φ ∘ (id ⊗₁ φ) ∘ α→ _ _ _ ∘ ((η _ ⊗₁ η _) ⊗₁ id)          ≡⟨ refl⟩∘⟨ refl⟩∘⟨ associator .Isoⁿ.to ._=>_.is-natural _ _ _ ⟩
      φ ∘ (id ⊗₁ φ) ∘ (η _ ⊗₁ (η _ ⊗₁ id)) ∘ α→ _ _ _          ≡⟨ refl⟩∘⟨ ⊗.pulll (idl _ ,ₚ refl) ⟩
      φ ∘ (η _ ⊗₁ (φ ∘ (η _ ⊗₁ id))) ∘ α→ _ _ _                ≡⟨ pushr (⊗.pushl (sym (idr _) ,ₚ sym (idl _))) ⟩
      (φ ∘ (η _ ⊗₁ id)) ∘ (id ⊗₁ (φ ∘ (η _ ⊗₁ id))) ∘ α→ _ _ _ ∎
    l .left-strength-η =
      (φ ∘ (η _ ⊗₁ id)) ∘ (id ⊗₁ η _) ≡⟨ pullr (⊗.collapse (idr _ ,ₚ idl _)) ⟩
      φ ∘ (η _ ⊗₁ η _)                ≡⟨ unit-φ ⟩
      η _                             ∎
    l .left-strength-μ =
      (φ ∘ (η _ ⊗₁ id)) ∘ (id ⊗₁ μ _)                      ≡⟨ pullr (⊗.collapse (idr _ ,ₚ idl _)) ⟩
      φ ∘ (η _ ⊗₁ μ _)                                     ≡˘⟨ refl⟩∘⟨ ⊗.collapse3 (cancell μ-idl ,ₚ elimr (eliml M-id)) ⟩
      φ ∘ (μ _ ⊗₁ μ _) ∘ (M₁ (η _) ⊗₁ M₁ id) ∘ (η _ ⊗₁ id) ≡⟨ pulll mult-φ ⟩
      (μ _ ∘ M₁ φ ∘ φ) ∘ (M₁ (η _) ⊗₁ M₁ id) ∘ (η _ ⊗₁ id) ≡⟨ pullr (pullr (extendl (φ.is-natural _ _ _))) ⟩
      μ _ ∘ M₁ φ ∘ M₁ (η _ ⊗₁ id) ∘ φ ∘ (η _ ⊗₁ id)        ≡⟨ refl⟩∘⟨ M.pulll refl ⟩
      μ _ ∘ M₁ (φ ∘ (η _ ⊗₁ id)) ∘ φ ∘ (η _ ⊗₁ id)         ∎
```
</details>

<!--
```agda
module _ {o ℓ}
  {C : Precategory o ℓ} (Cᵐ : Monoidal-category C) {M : Functor C C}
  (monad : Monad-on M)
  where
  open Cat.Reasoning C
  open Cat.Monoidal.Functor Cᵐ Cᵐ
  open Monoidal-category Cᵐ
  open Monad-on monad
  private
    module M = Cat.Functor.Reasoning M
    module M² = Cat.Functor.Reasoning (M F∘ M)
  open is-iso
```
-->

[[Reversing|reverse monoidal category]] the construction, we also get a
right strength.

```agda
  monoidal≃commutative
    : Monoidal-monad-on Cᵐ monad
    ≃ Σ (Monad-strength Cᵐ monad) is-commutative-strength
  monoidal≃commutative = Iso→Equiv is where
    is : Iso _ _
    is .fst m = s , s-comm module to-strength where
      open Monoidal-monad-on m
      open Monad-strength
      s : Monad-strength Cᵐ monad
      s .strength-left = monoidal→left-strength m
      s .strength-right = monad-strength^rev .fst
        (monoidal→left-strength (monoidal-monad^rev m))
```

<details>
<summary>
The coherence of the strengths is tedious; it involves the naturality of
the associator and the coherence of the monoidal multiplication with the
associator.
</summary>

~~~{.quiver}
\[\begin{tikzcd}[column sep=small]
  {(A\otimes MB)\otimes C} &&& {A\otimes (MB\otimes C)} \\
  {(MA\otimes MB)\otimes C} &&& {A\otimes (MB\otimes MC)} \\
  {M(A\otimes B)\otimes C} & {(MA\otimes MB)\otimes MC} & {MA\otimes (MB\otimes MC)} & {A\otimes M(B\otimes C)} \\
  {M(A\otimes B)\otimes MC} &&& {MA\otimes M(B\otimes C)} \\
  {M((A\otimes B)\otimes C)} &&& {M(A\otimes (B\otimes C))}
  \arrow[from=1-1, to=1-4]
  \arrow[from=5-1, to=5-4]
  \arrow[from=1-1, to=2-1]
  \arrow[from=2-1, to=3-1]
  \arrow[from=1-4, to=2-4]
  \arrow[from=2-4, to=3-4]
  \arrow[from=3-1, to=4-1]
  \arrow[from=4-1, to=5-1]
  \arrow[from=3-4, to=4-4]
  \arrow[from=4-4, to=5-4]
  \arrow[from=2-1, to=3-2]
  \arrow[from=3-2, to=3-3]
  \arrow[from=3-3, to=4-4]
  \arrow[from=2-4, to=3-3]
  \arrow[from=3-2, to=4-1]
\end{tikzcd}\]
~~~

```agda
      s .strength-α→ =
        M₁ (α→ _ _ _) ∘ (φ ∘ (id ⊗₁ η _)) ∘ ((φ ∘ (η _ ⊗₁ id)) ⊗₁ id) ≡⟨ refl⟩∘⟨ pullr (⊗.weave (idl _ ,ₚ id-comm)) ⟩
        M₁ (α→ _ _ _) ∘ φ ∘ (φ ⊗₁ id) ∘ ((η _ ⊗₁ id) ⊗₁ η _)          ≡⟨ extendl3 M-α→ ⟩
        φ ∘ (id ⊗₁ φ) ∘ α→ _ _ _ ∘ ((η _ ⊗₁ id) ⊗₁ η _)               ≡⟨ refl⟩∘⟨ refl⟩∘⟨ associator .Isoⁿ.to ._=>_.is-natural _ _ _ ⟩
        φ ∘ (id ⊗₁ φ) ∘ (η _ ⊗₁ (id ⊗₁ η _)) ∘ α→ _ _ _               ≡⟨ pushr (extendl (⊗.weave (id-comm-sym ,ₚ sym (idl _)))) ⟩
        (φ ∘ (η _ ⊗₁ id)) ∘ (id ⊗₁ (φ ∘ (id ⊗₁ η _))) ∘ α→ _ _ _      ∎
```
</details>

The crucial part is the proof that the strength thus defined is
[[commutative|commutative monad]].
Recall that this means that the two monoidal multiplications induced by
the strength agree; in fact, we will show that they are both equal to
$\varphi$!

For the left-to-right composition, this is witnessed by the commutativity
of the following diagram; the other direction is completely symmetric.

~~~{.quiver}
\[\begin{tikzcd}
  {MA\otimes MB} && {MA \otimes MB} \\
  {MA\otimes M^2 B} & {M^2 A\otimes M^2 B} \\
  {M(A\otimes MB)} \\
  {M(MA\otimes MB)} \\
  {M^2(A\otimes B)} && {M(A\otimes B)}
  \arrow["{MA \otimes \eta_{MB}}"', from=1-1, to=2-1]
  \arrow["\varphi"', from=2-1, to=3-1]
  \arrow["\mu"', from=5-1, to=5-3]
  \arrow["{M(\eta_A \otimes MB)}"', from=3-1, to=4-1]
  \arrow["M\varphi"', from=4-1, to=5-1]
  \arrow["{M\eta_A\otimes M^2 B}", from=2-1, to=2-2]
  \arrow["\varphi", from=2-2, to=4-1]
  \arrow["{\mu_A\otimes \mu_B}"', from=2-2, to=1-3]
  \arrow["\varphi", from=1-3, to=5-3]
  \arrow[Rightarrow, no head, from=1-1, to=1-3]
\end{tikzcd}\]
~~~

```agda
      opaque
        left≡φ : left-φ s ≡ M-mult
        left≡φ = ext λ (A , B) →
          μ _ ∘ M₁ (φ ∘ (η _ ⊗₁ id)) ∘ φ ∘ (id ⊗₁ η _)       ≡⟨ refl⟩∘⟨ M.popr (extendl (sym (φ.is-natural _ _ _))) ⟩
          μ _ ∘ M₁ φ ∘ φ ∘ (M₁ (η _) ⊗₁ M₁ id) ∘ (id ⊗₁ η _) ≡⟨ pushr (pushr (refl⟩∘⟨ ⊗.collapse (elimr refl ,ₚ M.eliml refl))) ⟩
          (μ _ ∘ M₁ φ ∘ φ) ∘ (M₁ (η _) ⊗₁ η _)               ≡˘⟨ pulll mult-φ ⟩
          φ ∘ (μ _ ⊗₁ μ _) ∘ (M₁ (η _) ⊗₁ η _)               ≡⟨ elimr (⊗.annihilate (μ-idl ,ₚ μ-idr)) ⟩
          φ                                                  ∎

        right≡φ : right-φ s ≡ M-mult
        right≡φ = ext λ (A , B) →
          μ _ ∘ M₁ (φ ∘ (id ⊗₁ η _)) ∘ φ ∘ (η _ ⊗₁ id)       ≡⟨ refl⟩∘⟨ M.popr (extendl (sym (φ.is-natural _ _ _))) ⟩
          μ _ ∘ M₁ φ ∘ φ ∘ (M₁ id ⊗₁ M₁ (η _)) ∘ (η _ ⊗₁ id) ≡⟨ pushr (pushr (refl⟩∘⟨ ⊗.collapse (M.eliml refl ,ₚ elimr refl))) ⟩
          (μ _ ∘ M₁ φ ∘ φ) ∘ (η _ ⊗₁ M₁ (η _))               ≡˘⟨ pulll mult-φ ⟩
          φ ∘ (μ _ ⊗₁ μ _) ∘ (η _ ⊗₁ M₁ (η _))               ≡⟨ elimr (⊗.annihilate (μ-idr ,ₚ μ-idl)) ⟩
          φ                                                  ∎

        s-comm : is-commutative-strength s
        s-comm = right≡φ ∙ sym left≡φ
```

### Monoidal monads from commutative monads

In the other direction, we are given a commutative strength and we must
construct a monoidal monad. We already [[know|monoidal functors from
strong monads]] how to construct a monoidal *functor* structure on a
strong monad; all that remains is to check that it makes the monad
structure monoidal.

```agda
    is .snd .inv (s , s-comm) = m where
      open Monad-strength s
      open Monoidal-monad-on
      open Lax-monoidal-functor-on

      m : Monoidal-monad-on Cᵐ monad
      m .monad-monoidal = strength→monoidal s
```

The monoidal unit is $\eta_1$ by definition.

```agda
      m .unit-ε = refl
```

<details>
<summary>
The `unit-φ`{.Agda} coherence is not very interesting.
</summary>

~~~{.quiver}
\[\begin{tikzcd}
  {A\otimes B} && {MA\otimes MB} \\
  & {A\otimes MB} & {M(A\otimes MB)} \\
  && {M²(A\otimes B)} \\
  {M(A \otimes B)} && {M(A\otimes B)}
  \arrow["{\eta_A \otimes \eta_B}", from=1-1, to=1-3]
  \arrow["\tau", from=1-3, to=2-3]
  \arrow["M\sigma", from=2-3, to=3-3]
  \arrow["{A \otimes \eta_B}", from=1-1, to=2-2]
  \arrow["\eta", from=2-2, to=2-3]
  \arrow["\mu", from=3-3, to=4-3]
  \arrow["{\eta_{A\otimes B}}"', from=1-1, to=4-1]
  \arrow[Rightarrow, no head, from=4-1, to=4-3]
  \arrow["\sigma"', from=2-2, to=4-1]
  \arrow["\eta", from=4-1, to=3-3]
\end{tikzcd}\]
~~~

```agda
      m .unit-φ =
        (μ _ ∘ M₁ σ ∘ τ) ∘ (η _ ⊗₁ η _)            ≡⟨ pullr (pullr (refl⟩∘⟨ ⊗.expand (intror refl ,ₚ introl refl))) ⟩
        μ _ ∘ M₁ σ ∘ τ ∘ (η _ ⊗₁ id) ∘ (id ⊗₁ η _) ≡⟨ refl⟩∘⟨ refl⟩∘⟨ pulll right-strength-η ⟩
        μ _ ∘ M₁ σ ∘ η _ ∘ (id ⊗₁ η _)             ≡˘⟨ refl⟩∘⟨ extendl (unit.is-natural _ _ _) ⟩
        μ _ ∘ η _ ∘ σ ∘ (id ⊗₁ η _)                ≡⟨ cancell μ-idr ⟩
        σ ∘ (id ⊗₁ η _)                            ≡⟨ left-strength-η ⟩
        η _                                        ∎
```
</details>

As expected, the proof of `mult-φ`{.Agda} involves the commutativity of
the strength. At a very high level, we are trying to collapse
the sequence $\tau\sigma\tau\sigma$ to $\tau\sigma$: we first need
to commute the two strengths in the middle to obtain $\tau\tau\sigma
\sigma$, and then use the fact that two consecutive applications of the
strengths result in just one after commuting with $\mu$.
Concretely, this manifests as the following diagram, where the
commutation of the strengths is highlighted.

~~~{.quiver}
\[\begin{tikzcd}
  {M^2 A\otimes M^2 B} &&& {MA\otimes MB} \\
  {M(MA\otimes M^2 B)} && {MA\otimes M^2B} \\
  {M^2(MA\otimes MB)} & {M^2(A \otimes M^2 B)} \\
  {M(MA\otimes MB)} & {M^3(A \otimes MB)} & {M(A\otimes M^2 B)} \\
  &&& {M(A\otimes MB)} \\
  {M^2(A\otimes MB)} && {M^3(A\otimes B)} \\
  & {M(A \otimes MB)} \\
  {M^3(A\otimes B)} &&& {M^2(A\otimes B)} \\
  &&& {M(A\otimes B)}
  \arrow["{\mu\otimes \mu}", from=1-1, to=1-4]
  \arrow["\tau"', from=1-1, to=2-1]
  \arrow["M\sigma"', draw={rgb,255:red,214;green,92;blue,214}, from=2-1, to=3-1]
  \arrow["\mu"', from=3-1, to=4-1]
  \arrow["M\tau"', from=4-1, to=6-1]
  \arrow["{M^2\sigma}"', from=6-1, to=8-1]
  \arrow["\tau", from=1-4, to=5-4]
  \arrow["M\sigma", from=5-4, to=8-4]
  \arrow["\mu", from=8-4, to=9-4]
  \arrow["{M^2\tau}"', draw={rgb,255:red,214;green,92;blue,214}, from=3-1, to=4-2]
  \arrow["{(M)\mu}"', draw={rgb,255:red,214;green,92;blue,214}, from=4-2, to=6-1]
  \arrow["\mu"', from=6-1, to=7-2]
  \arrow["M\tau", draw={rgb,255:red,214;green,92;blue,214}, from=2-1, to=3-2]
  \arrow["{M^2\sigma}", draw={rgb,255:red,214;green,92;blue,214}, from=3-2, to=4-2]
  \arrow["{\mu\otimes MMB}"', from=1-1, to=2-3]
  \arrow["{MA\otimes \mu}"', from=2-3, to=1-4]
  \arrow["\mu", from=3-2, to=4-3]
  \arrow["M\sigma", from=4-3, to=6-1]
  \arrow["{M^2\sigma}", from=6-1, to=6-3]
  \arrow["{M(A\otimes\mu)}", from=4-3, to=5-4]
  \arrow["\tau", from=2-3, to=4-3]
  \arrow["{(M)\mu}", from=6-3, to=8-4]
  \arrow["M\sigma", from=7-2, to=8-4]
  \arrow["{(M)\mu}"', from=8-1, to=8-4]
\end{tikzcd}\]
~~~

The morphisms labelled $(M)\mu$ alternate between being $\mu$ and $M\mu$,
in a way that is allowed by the associativity law because they are
followed by $\mu$.

```agda
      m .mult-φ =
        (μ _ ∘ M₁ σ ∘ τ) ∘ (μ _ ⊗₁ μ _)                        ≡⟨ refl⟩∘⟨ ⊗.expand (M.introl refl ,ₚ intror refl) ⟩
        (μ _ ∘ M₁ σ ∘ τ) ∘ (M₁ id ⊗₁ μ _) ∘ (μ _ ⊗₁ id)        ≡⟨ pullr (pullr (extendl (τ.is-natural _ _ _))) ⟩
        μ _ ∘ M₁ σ ∘ M₁ (id ⊗₁ μ _) ∘ τ ∘ (μ _ ⊗₁ id)          ≡⟨ refl⟩∘⟨ refl⟩∘⟨ refl⟩∘⟨ right-strength-μ ⟩
        μ _ ∘ M₁ σ ∘ M₁ (id ⊗₁ μ _) ∘ μ _ ∘ M₁ τ ∘ τ           ≡⟨ refl⟩∘⟨ M.pulll (left-strength-μ ∙ assoc _ _ _) ⟩
        μ _ ∘ M₁ ((μ _ ∘ M₁ σ) ∘ σ) ∘ μ _ ∘ M₁ τ ∘ τ           ≡⟨ refl⟩∘⟨ extendl (M.popr (sym (mult.is-natural _ _ _))) ⟩
        μ _ ∘ M₁ (μ _ ∘ M₁ σ) ∘ (μ _ ∘ M₁ (M₁ σ)) ∘ M₁ τ ∘ τ   ≡⟨ extendl (M.popl μ-assoc) ⟩
        (μ _ ∘ μ _) ∘ M₁ (M₁ σ) ∘ (μ _ ∘ M₁ (M₁ σ)) ∘ M₁ τ ∘ τ ≡⟨ pullr (extendl (mult.is-natural _ _ _)) ⟩
        μ _ ∘ M₁ σ ∘ μ _ ∘ (μ _ ∘ M₁ (M₁ σ)) ∘ M₁ τ ∘ τ        ≡˘⟨ refl⟩∘⟨ refl⟩∘⟨ extendl (extendl μ-assoc) ⟩
        μ _ ∘ M₁ σ ∘ μ _ ∘ (M₁ (μ _) ∘ M₁ (M₁ σ)) ∘ M₁ τ ∘ τ   ≡⟨ refl⟩∘⟨ refl⟩∘⟨ refl⟩∘⟨ sym (assoc _ _ _) ∙ M.extendl3 (sym (s-comm ηₚ _)) ⟩
        μ _ ∘ M₁ σ ∘ μ _ ∘ M₁ (μ _) ∘ M₁ (M₁ τ) ∘ M₁ σ ∘ τ     ≡⟨ refl⟩∘⟨ refl⟩∘⟨ extendl μ-assoc ⟩
        μ _ ∘ M₁ σ ∘ μ _ ∘ μ _ ∘ M₁ (M₁ τ) ∘ M₁ σ ∘ τ          ≡⟨ refl⟩∘⟨ refl⟩∘⟨ refl⟩∘⟨ extendl (mult.is-natural _ _ _) ⟩
        μ _ ∘ M₁ σ ∘ μ _ ∘ M₁ τ ∘ μ _ ∘ M₁ σ ∘ τ               ≡˘⟨ refl⟩∘⟨ extendl (mult.is-natural _ _ _) ⟩
        μ _ ∘ μ _ ∘ M₁ (M₁ σ) ∘ M₁ τ ∘ μ _ ∘ M₁ σ ∘ τ          ≡˘⟨ extendl μ-assoc ⟩
        μ _ ∘ M₁ (μ _) ∘ M₁ (M₁ σ) ∘ M₁ τ ∘ μ _ ∘ M₁ σ ∘ τ     ≡⟨ refl⟩∘⟨ M.pulll3 refl ⟩
        μ _ ∘ M₁ (μ _ ∘ M₁ σ ∘ τ) ∘ μ _ ∘ M₁ σ ∘ τ             ∎
```

### Wrapping up

Finally, we check that these two transformations are mutual inverses.

Given a commutative strength, we must check that the round-trip defined
above yields the same left and right strengths we started with; the
situation isn't entirely symmetric because we've made a *choice* to
use the right strength first when defining the monoidal structure, but
both verifications are straightforward.

```agda
    is .snd .rinv (s , s-comm) = Σ-prop-path (λ _ → hlevel 1)
      (Monad-strength-path Cᵐ monad
        (Left-monad-strength-path Cᵐ monad
          (Left-strength-path Cᵐ M (sym l)))
        (Right-monad-strength-path Cᵐ monad
          (Right-strength-path Cᵐ M (sym r))))
      where
        open Monad-strength s
        l : left-strength ≡ is .fst (is .snd .inv (s , s-comm)) .fst .Monad-strength.left-strength
        l = ext λ (A , B) →
          σ                              ≡⟨ insertl μ-idr ⟩
          μ _ ∘ η _ ∘ σ                  ≡⟨ refl⟩∘⟨ unit.is-natural _ _ _ ⟩
          μ _ ∘ M₁ σ ∘ η _               ≡˘⟨ pullr (pullr right-strength-η) ⟩
          (μ _ ∘ M₁ σ ∘ τ) ∘ (η _ ⊗₁ id) ∎
        r : right-strength ≡ is .fst (is .snd .inv (s , s-comm)) .fst .Monad-strength.right-strength
        r = ext λ (A , B) →
          τ                                     ≡⟨ insertl μ-idl ⟩
          μ _ ∘ M₁ (η _) ∘ τ                    ≡˘⟨ refl⟩∘⟨ M.pulll left-strength-η ⟩
          μ _ ∘ M₁ σ ∘ M₁ (id ⊗₁ η _) ∘ τ       ≡˘⟨ pullr (pullr (τ.is-natural _ _ _)) ⟩
          (μ _ ∘ M₁ σ ∘ τ) ∘ (⌜ M₁ id ⌝ ⊗₁ η _) ≡⟨ ap! M-id ⟩
          (μ _ ∘ M₁ σ ∘ τ) ∘ (id ⊗₁ η _)        ∎
```

For the other round-trip, we've *already* proved above that we get the
same $\varphi$ we started with; moreover, the monoidal unit $\epsilon$
becomes $\eta_1$, but the axioms of a monoidal monad force those to be
the same, so we're done.

```agda
    is .snd .linv m = Monoidal-monad-on-path Cᵐ
      (Lax-monoidal-functor-on-path
        (sym unit-ε)
        (to-strength.left≡φ m))
      where open Monoidal-monad-on m
```

<!--
```agda
  module monoidal≃commutative = Equiv monoidal≃commutative
```
-->

### Symmetric monoidal monads and commutative symmetric strengths {defines="symmetric-monoidal-monad"}

We can now refine this to Kock's [-@Kock:monoidal-monads] original
result, which concerns *symmetric* monoidal monads in a [[symmetric
monoidal category]].

<!--
```agda
  module _ (Cˢ : Symmetric-monoidal Cᵐ) where
    open Symmetric-monoidal Cˢ
```
-->

We define a **symmetric monoidal monad** as a monoidal monad whose
underlying monoidal functor is [[symmetric|symmetric monoidal functor]].

```agda
    is-symmetric-monoidal-monad : Monoidal-monad-on Cᵐ monad → Type (o ⊔ ℓ)
    is-symmetric-monoidal-monad m = is-symmetric-functor Cˢ Cˢ
      (_ , m .Monoidal-monad-on.monad-monoidal)
```

Then, we have that, *[[over|equivalence over]]* the above equivalence
between monoidal monads and commutative strengths, the property of being
a *symmetric* monoidal monad is equivalent to the property of being a
*symmetric* strength.

Given a symmetric monoidal monad, we immediately see that the induced
left and right strengths are related by the braiding.

<!--
```agda
    module _ (m : Monoidal-monad-on Cᵐ monad) where
      open Monoidal-monad-on m
```
-->

```agda
      symmetric-monoidal→symmetric-strength
        : is-symmetric-monoidal-monad m
        → is-symmetric-monad-strength Cᵇ (monoidal≃commutative.to m .fst)
      symmetric-monoidal→symmetric-strength sy =
        (φ ∘ (id ⊗₁ η _)) ∘ β→  ≡⟨ pullr (sym (β→.is-natural _ _ _)) ⟩
        φ ∘ β→ ∘ (η _ ⊗₁ id)    ≡⟨ extendl sy ⟩
        M₁ β→ ∘ φ ∘ (η _ ⊗₁ id) ∎
```

Now, given a symmetric commutative strength, we can use the commutativity
as follows to conclude that the induced monoidal functor is symmetric:

~~~{.quiver}
\[\begin{tikzcd}
  {MA \otimes MB} && {MB \otimes MA} \\
  {M(A \otimes MB)} & {M(MA \otimes B)} & {M (B \otimes MA)} \\
  {M^2 (A \otimes B)} && {M^2 (B \otimes A)} \\
  {M (A \otimes B)} && {M (B \otimes A)}
  \arrow["\beta", from=1-1, to=1-3]
  \arrow["\tau"', from=1-1, to=2-1]
  \arrow["M\sigma"', from=2-1, to=3-1]
  \arrow["\mu"', from=3-1, to=4-1]
  \arrow["\tau", from=1-3, to=2-3]
  \arrow["M\sigma", from=2-3, to=3-3]
  \arrow["\mu", from=3-3, to=4-3]
  \arrow["M\beta"', from=4-1, to=4-3]
  \arrow["\sigma"', from=1-1, to=2-2]
  \arrow["M\tau"', from=2-2, to=3-1]
  \arrow["M\beta", from=2-2, to=2-3]
  \arrow["{M^2\beta}", from=3-1, to=3-3]
\end{tikzcd}\]
~~~

<!--
```agda
    module _ ((s , s-comm) : Σ _ is-commutative-strength) where
      open Monad-strength s
```
-->

```agda
      symmetric-strength→symmetric-monoidal
        : is-symmetric-monad-strength Cᵇ s
        → is-symmetric-monoidal-monad (monoidal≃commutative.from (s , s-comm))
      symmetric-strength→symmetric-monoidal sy =
        (μ _ ∘ M₁ σ ∘ τ) ∘ β→       ≡⟨ pullr (pullr sy) ⟩
        μ _ ∘ M₁ σ ∘ M₁ β→ ∘ σ      ≡˘⟨ refl⟩∘⟨ M.extendl (swizzle sy has-is-symmetric (M.annihilate has-is-symmetric)) ⟩
        μ _ ∘ M₁ (M₁ β→) ∘ M₁ τ ∘ σ ≡⟨ extendl (mult.is-natural _ _ _) ⟩
        M₁ β→ ∘ μ _ ∘ M₁ τ ∘ σ      ≡⟨ refl⟩∘⟨ s-comm ηₚ _ ⟩
        M₁ β→ ∘ μ _ ∘ M₁ σ ∘ τ      ∎
```

Packaging all of this together, we conclude with the desired equivalence
between the *structure* of a symmetric monoidal monad and the *structure*
of a symmetric commutative strength.

```agda
    symmetric≃symmetric
      : is-symmetric-monoidal-monad ≃[ monoidal≃commutative ]
        (is-symmetric-monad-strength Cᵇ ⊙ fst)
    symmetric≃symmetric = prop-over-ext monoidal≃commutative (hlevel 1) (hlevel 1)
      symmetric-monoidal→symmetric-strength symmetric-strength→symmetric-monoidal

    symmetric-monoidal≃symmetric-commutative
      : Σ (Monoidal-monad-on Cᵐ monad) is-symmetric-monoidal-monad
      ≃ Σ (Monad-strength Cᵐ monad) (λ s →
        is-commutative-strength s × is-symmetric-monad-strength Cᵇ s)
    symmetric-monoidal≃symmetric-commutative =
      over→total monoidal≃commutative symmetric≃symmetric
      ∙e Σ-assoc e⁻¹
```
