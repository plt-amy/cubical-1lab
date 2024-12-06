---
description: |
  F-Algebras, Lambek's lemma, Adámek's theorem, free monads and
  free relative extension systems.
---
<!--
```agda
open import Cat.Functor.FullSubcategory
open import Cat.Diagram.Monad.Relative
open import Cat.Functor.Adjoint.Monad
open import Cat.Diagram.Colimit.Base
open import Cat.Functor.Equivalence
open import Cat.Diagram.Initial
open import Cat.Displayed.Total
open import Cat.Functor.Adjoint
open import Cat.Displayed.Base
open import Cat.Diagram.Monad
open import Cat.Prelude

open import Data.Nat

open import Order.Instances.Nat
open import Order.Cat

import Cat.Functor.Reasoning
import Cat.Reasoning
```
-->
```agda
module Cat.Functor.Algebra where
```

```agda
module _ {o ℓ} {C : Precategory o ℓ} (F : Functor C C) where
```

# Algebras over an endofunctor {defines="f-algebra algebra-over-an-endofunctor"}

<!--
```agda
  private module F = Cat.Functor.Reasoning F
  open Cat.Reasoning C
  open Displayed
  open Total-hom
```
-->

Let $F : \cC \to \cC$ be an arbitrary endofunctor on $\cC$. We can view
$F$ as a means of embellishing an object with extra structure: for
instance, the functor $X \mapsto X + 1$ endows $X$ with an additional
point and an endomap $X \to X$, the functor $X \mapsto A \times X + 1$
adds an extra point and a copy of $A$, etc. Consequently, a map $\cC(F(A), A)$
picks out a suitably structured $A$; e.g. writing a map $\cC(A + 1, A)$
requires $A$ to be equipped with a global element and an endomap $\cC(A,A)$.
In analogy to [[monad algebras]], a map $\cC(F(A),A)$ is called an
**F-algebra** on $A$^[Alternatively, we can view $F$-algebras as monad
algebras over functors that lack algebraic structure.].

Likewise, a map $f : \cC(A,B)$ between two $F$-algebras $\alpha : \cC(F(A),A)$
and $\beta : \cC(F(B), B)$ is an **F-algebra homomorphism** if it commutes
with the algebras on $A$ and $B$, as in the following digram.

~~~{.quiver}
\begin{tikzcd}
  {F(A)} && {F(B)} \\
  \\
  A && B
  \arrow["\alpha"', from=1-1, to=3-1]
  \arrow["f"', from=3-1, to=3-3]
  \arrow["\beta", from=1-3, to=3-3]
  \arrow["{F(f)}", from=1-1, to=1-3]
\end{tikzcd}
~~~

Intuitively, $F$-algebra morphisms are morphisms that preserve the structure
encoded by $F$; going back to our previous example, an $F$-algebra homomorphism
between $\alpha : A + 1 \to A$ and $\beta : B + 1 \to B$ is a map
$f : A \to B$ that preserves the designated global elements of $A$ and $B$
and commutes with the chosen endomaps.

We can assemble $F$-algebras into a displayed category over $\cC$: the
type of objects over $A$ consists of all possible algebra structures on
$A$, and the type of morphisms over $f : \cC(A,B)$ are proofs that $f$ is an
$F$-algebra homomorphism.

```agda
  F-Algebras : Displayed C ℓ ℓ
  F-Algebras .Ob[_] a = Hom (F.₀ a) a
  F-Algebras .Hom[_] f α β = f ∘ α ≡ β ∘ F.₁ f
  F-Algebras .Hom[_]-set _ _ _ = hlevel 2
  F-Algebras .id' = idl _ ∙ intror F.F-id
  F-Algebras ._∘'_ p q = pullr q ∙ pulll p ∙ pullr (sym (F.F-∘ _ _))
  F-Algebras .idr' _ = prop!
  F-Algebras .idl' _ = prop!
  F-Algebras .assoc' _ _ _ = prop!
```

We can take the [[total category]] of this displayed category to recover
the more traditional category of $F$-algebras.

```agda
  FAlg : Precategory (o ⊔ ℓ) ℓ
  FAlg = ∫ F-Algebras

  module FAlg = Cat.Reasoning FAlg

  F-Algebra : Type _
  F-Algebra = FAlg.Ob
```

## Lambek's lemma {defines="lambeks-lemma"}

As noted above, $F$-algebras are a tool for picking out objects equipped
with a semantics for a "signature" described by $F$. This leads to a very
natural question: can we characterize the syntax generated from the
signature of $F$?

To answer this question, we must pin down exactly what we mean by "syntax".
Categorically, we have two options: [[initial objects]], and
[[free objects]]; these correspond to "syntax" and "syntax with generators",
respectively. We will focus initial $F$-algebras for now: free objects are
much harder to come by, and initial $F$-algebras have a nice characterization:
they are the *least fixpoints* of functors. This result is known as Lambek's
lemma, and we shall prove it now.

First, a tiny lemma. Let $(A, \alpha)$ be an $F$-algebra, and note that
$(F(A), F(\alpha))$ is also an $F$-algebra. If $\alpha$ has a section
$f : (A, \alpha) \to (F(A), F(\alpha))$, then $f$ and $\alpha$ are
inverses.

```agda
  algebra-section→inverses
    : ∀ {a} (α : Hom (F.₀ a) a)
    → (f : FAlg.Hom (a , α) (F.₀ a , F.₁ α))
    → f .hom section-of α
    → Inverses α (f .hom)
  algebra-section→inverses α f section =
    make-inverses section $
      f .hom ∘ α           ≡⟨ f .preserves ⟩
      F.₁ α ∘ F.₁ (f .hom) ≡⟨ F.annihilate section ⟩
      id                   ∎
```

On to the main result. Let $(A, \alpha)$ be an initial $F$-algebra. As before,
$(F(A), F(\alpha))$ is also an $F$-algebra, so initiality gives us a unique
$F$-algebra morphism $\textrm{unroll} : (A, \alpha) \to (F(A), F(\alpha))$.
Likewise, $\alpha$ induces an $F$-algebra morphism $\textrm{roll} : (F(A), F(\alpha) \to (A, \alpha))$.
Furthermore, $\textrm{unroll}$ is a section of $\textrm{roll}$, as maps
out of the initial object are unique. Therefore, $\textrm{unroll}$ must
also be an inverse, so $\alpha$ is invertible.

```agda
  lambek : ∀ {a} (α : Hom (F.₀ a) a) → is-initial FAlg (a , α) → is-invertible α
  lambek {a} α initial = inverses→invertible $
    algebra-section→inverses α unroll roll-unroll
    where
      unroll : FAlg.Hom (a , α) (F.₀ a , F.₁ α)
      unroll = initial (F.₀ a , F.₁ α) .centre

      roll : FAlg.Hom (F.₀ a , F.₁ α) (a , α)
      roll .hom = α
      roll .preserves = refl

      roll-unroll : α ∘ unroll .hom ≡ id
      roll-unroll =
        ap hom $
        is-contr→is-prop (initial (a , α)) (roll FAlg.∘ unroll) FAlg.id
```

This result means that an initial $F$-algebra $(A, \alpha)$ is a fixpoint of the
functor $F$, and should be thought of as the the object $F(F(F(\cdots)))$.
The canonical example is the initial algebra of $X \mapsto 1 + X$: if it
exists, this will be an object of the form $1 + (1 + (1 + \cdots))$: in
$\Sets$, this initial algebra is the natural numbers! In general, initial
$F$-algebras are how we give a categorical semantics to non-indexed
inductive datatypes like `Nat`{.Agda} or `List`{.Agda}.

## Adámek's fixpoint theorem {defines="adameks-theorem"}

Note that initial $F$-algebras need not exist for a given functor $F$.
Nevertheless, **Adámek's fixpoint theorem** lets us construct some
initial $F$-algebras, provided that:

1. $\cC$ has an initial object.
2. $\cC$ has a colimit for the diagram:
~~~{.quiver}
\begin{tikzcd}
  \bot && {F(\bot)} && {F(F(\bot))} && \cdots
  \arrow["{!}", from=1-1, to=1-3]
  \arrow["{F(!)}", from=1-3, to=1-5]
  \arrow[from=1-5, to=1-7]
\end{tikzcd}
~~~
3. $F$ preserves the aforementioned colimit.

Note that this construction is precisely a categorified version of
[[Kleene's fixpoint theorem]], so we will be following the exact same
proof strategy.

We begin by constructing the above diagram as a functor $F^{(-)}(\bot) : \omega \to \cC$,
where $\omega$ is the [[poset of natural numbers]], regarded as a category.

<!--
```agda
  private
    ω : Precategory _ _
    ω = poset→category Nat-poset

  module _
    (initial : Initial C)
    where

    open Functor
    open Initial initial
```
-->

```agda
    F₀ⁿ[⊥] : Nat → Ob
    F₀ⁿ[⊥] zero = bot
    F₀ⁿ[⊥] (suc n) = F.₀ (F₀ⁿ[⊥] n)

    F₁ⁿ[⊥] : ∀ {m n} → m ≤ n → Hom (F₀ⁿ[⊥] m) (F₀ⁿ[⊥] n)
    F₁ⁿ[⊥] 0≤x = ¡
    F₁ⁿ[⊥] (s≤s p) = F.₁ (F₁ⁿ[⊥] p)

    Fⁿ[⊥] : Functor ω C
```

<details>
<summary>Functoriality follows from some straightforward induction.
</summary>

```agda
    Fⁿ[⊥]-id : ∀ n → F₁ⁿ[⊥] (≤-refl {n}) ≡ id
    Fⁿ[⊥]-id zero = ¡-unique id
    Fⁿ[⊥]-id (suc n) = F.elim (Fⁿ[⊥]-id n)

    Fⁿ[⊥]-∘
      : ∀ {m n o}
      → (p : n ≤ o) (q : m ≤ n)
      → F₁ⁿ[⊥] (≤-trans q p) ≡ F₁ⁿ[⊥] p ∘ F₁ⁿ[⊥] q
    Fⁿ[⊥]-∘ p 0≤x = ¡-unique₂ _ _
    Fⁿ[⊥]-∘ (s≤s p) (s≤s q) = F.expand (Fⁿ[⊥]-∘ p q)

    Fⁿ[⊥] .F₀ = F₀ⁿ[⊥]
    Fⁿ[⊥] .F₁ = F₁ⁿ[⊥]
    Fⁿ[⊥] .F-id {n} = Fⁿ[⊥]-id n
    Fⁿ[⊥] .F-∘ p q = Fⁿ[⊥]-∘ p q
```
</details>

Next, note that every $F$-algebra $(A, \alpha)$ can be extended to
a morphism $\cC(F^{n}(\bot), A)$. Furthermore, this extension is natural
in $n$.

```agda
    Fⁿ[⊥]-fold : ∀ {a} → Hom (F.₀ a) a → ∀ n → Hom (F₀ⁿ[⊥] n) a
    Fⁿ[⊥]-fold α zero = ¡
    Fⁿ[⊥]-fold α (suc n) = α ∘ F.₁ (Fⁿ[⊥]-fold α n)

    Fⁿ[⊥]-fold-nat
      : ∀ {a} {α : Hom (F.₀ a) a} {m n}
      → (m≤n : m ≤ n)
      → Fⁿ[⊥]-fold α n ∘ F₁ⁿ[⊥] m≤n ≡ Fⁿ[⊥]-fold α m
    Fⁿ[⊥]-fold-nat 0≤x = sym (¡-unique _)
    Fⁿ[⊥]-fold-nat (s≤s m≤n) = F.pullr (Fⁿ[⊥]-fold-nat m≤n)
```

Now, suppose that $\cC$ has a colimit $F^{\infty}(\bot)$ of the diagram
$F^{(-)}(\bot)$, and $F$ preserves this colimit. We can equip $F^{\infty}(\bot)$
with an $F$-algebra $\mathrm{roll} : F(F^{\infty}(\bot)) \to F^{\infty}(\bot)$
by using the universal property of the colimit $F(F^{\infty}(\bot))$.

```agda
    adamek : Colimit Fⁿ[⊥] → preserves-colimit F Fⁿ[⊥] → Initial FAlg
    adamek Fⁿ[⊥]-colim F-pres-Fⁿ[⊥]-colim = ∐Fⁿ[⊥]-initial
      module adamek where
      module ∐Fⁿ[⊥] = Colimit Fⁿ[⊥]-colim
      module F[∐Fⁿ[⊥]] = is-colimit (F-pres-Fⁿ[⊥]-colim ∐Fⁿ[⊥].has-colimit)

      ∐Fⁿ[⊥] : Ob
      ∐Fⁿ[⊥] = ∐Fⁿ[⊥].coapex

      roll : Hom (F.₀ ∐Fⁿ[⊥]) ∐Fⁿ[⊥]
      roll = F[∐Fⁿ[⊥]].universal (∐Fⁿ[⊥].ψ ⊙ suc) (∐Fⁿ[⊥].commutes ⊙ s≤s)
```

Next, we can extend any F-algebra $(A, \alpha)$ to a morphism $F^{\infty}(\bot)$
by using the universal property, along with the extensions $\cC(F^{n}(\bot), A)$
we constructed earlier.

```agda
      fold : ∀ {a} → Hom (F.₀ a) a → Hom ∐Fⁿ[⊥] a
      fold α = ∐Fⁿ[⊥].universal (Fⁿ[⊥]-fold α) Fⁿ[⊥]-fold-nat
```

This extension commutes with $\mathrm{roll}$, and thus induces an
$F$-algebra morphism $(F^{\infty}(\bot), \mathrm{roll}) \to (A, \alpha)$.

```agda
      fold-roll : ∀ {a} (α : Hom (F.₀ a) a) → fold α ∘ roll ≡ α ∘ F.₁ (fold α)
      fold-roll α =
        F[∐Fⁿ[⊥]].unique₂ (Fⁿ[⊥]-fold α ⊙ suc) (Fⁿ[⊥]-fold-nat ⊙ s≤s)
          (λ j →
            (fold α ∘ roll) ∘ F.₁ (∐Fⁿ[⊥].ψ j) ≡⟨ pullr (F[∐Fⁿ[⊥]].factors _ _) ⟩
            fold α ∘ ∐Fⁿ[⊥].ψ (suc j)          ≡⟨ ∐Fⁿ[⊥].factors _ _ ⟩
            Fⁿ[⊥]-fold α (suc j)               ∎)
          (λ j →
            (α ∘ F.₁ (fold α)) ∘ F.₁ (∐Fⁿ[⊥].ψ j) ≡⟨ F.pullr (∐Fⁿ[⊥].factors _ _) ⟩
            α ∘ F.F₁ (Fⁿ[⊥]-fold α j)             ∎)
```

Furthermore, this extension is unique: this follows from a quick induction
paired with the universal property of $F^{\infty}(\bot)$.

```agda
      fold-step
        : ∀ {a} {α : Hom (F.₀ a) a}
        → (f : Hom ∐Fⁿ[⊥] a)
        → f ∘ roll ≡ α ∘ F.₁ f
        → ∀ n → f ∘ ∐Fⁿ[⊥].ψ n ≡ Fⁿ[⊥]-fold α n
      fold-step {α = α} f p zero = sym (¡-unique _)
      fold-step {α = α} f p (suc n) =
         f ∘ ∐Fⁿ[⊥].ψ (suc n)               ≡˘⟨ ap (f ∘_) (F[∐Fⁿ[⊥]].factors _ _) ⟩
         f ∘ roll ∘ F.F₁ (∐Fⁿ[⊥].ψ n)       ≡⟨ pulll p ⟩
         (α ∘ F.F₁ f) ∘ F.F₁ (∐Fⁿ[⊥].ψ n)   ≡⟨ F.pullr (fold-step f p n) ⟩
         α ∘ F.₁ (Fⁿ[⊥]-fold _ n)           ∎

      fold-unique
        : ∀ {a} {α : Hom (F.₀ a) a}
        → (f : Hom ∐Fⁿ[⊥] a)
        → f ∘ roll ≡ α ∘ F.₁ f
        → fold α ≡ f
      fold-unique f p = sym $ ∐Fⁿ[⊥].unique _ _ _ (fold-step f p)
```

If we put all the pieces together, we observe that $(F^{\infty}(\bot), \mathrm{roll})$
is an initial $F$-algebra.

```agda
      ∐Fⁿ[⊥]-initial : Initial FAlg
      ∐Fⁿ[⊥]-initial .Initial.bot .fst = ∐Fⁿ[⊥]
      ∐Fⁿ[⊥]-initial .Initial.bot .snd = roll
      ∐Fⁿ[⊥]-initial .Initial.has⊥ (a , α) .centre .hom = fold α
      ∐Fⁿ[⊥]-initial .Initial.has⊥ (a , α) .centre .preserves = fold-roll α
      ∐Fⁿ[⊥]-initial .Initial.has⊥ (a , α) .paths f =
        total-hom-path F-Algebras (fold-unique (f .hom) (f .preserves)) prop!
```

## Free algebras and free monads

<!-- [TODO: Reed M, 31/03/2024] Migrate to free monad page once it is written. -->

In the previous section, we dismissed free $F$-algebras as somewhat rare
objects. It is now time to see *why* this is the case. Suppose that
$\cC$ has all free $F$-algebras: this is equivalent to requiring a left
adjoint to the functor that forgets $F$-algebras.

```agda
  module _ {Free : Functor C FAlg} (Free⊣π : Free ⊣ πᶠ F-Algebras) where
    private
      module Free = Cat.Functor.Reasoning Free
      open _⊣_ Free⊣π
      open Functor
```

This adjunction [[induces a monad|monad-from-adjunction]] on $\cC$, which
we will call the **algebraically free monad** on $F$.

```agda
    Alg-free-monad : Monad-on _
    Alg-free-monad = Adjunction→Monad Free⊣π
```

That was pretty abstract, so let's unpack the data we have on hand.
The left adjoint takes each object $A$ to an object $F^{*}(A)$ that is equipped
with an $F$-algebra $\mathrm{roll} : \cC(F(F^{*}(A)),F^{*}(A))$.

<!--
```agda
    open Algebra-on {M = Alg-free-monad}
```
-->

```agda
    private
      F* : Ob → Ob
      F* x = Free.₀ x .fst

      roll : ∀ (x : Ob) → Hom (F.₀ (F* x)) (F* x)
      roll x = Free.₀ x .snd
```

Furthermore, the left adjoint takes each $f : \cC(A,B)$ to an $F$-algebra
homomorphism $F^{*}(f)$, as in the following diagram:

~~~{.quiver}
\begin{tikzcd}
  {F(F^*(A))} && {F(F^*(B))} \\
  \\
  {F^*(A)} && {F^*(B)}
  \arrow["{\mathrm{roll}_A}"', from=1-1, to=3-1]
  \arrow["{F^*(f)}"', from=3-1, to=3-3]
  \arrow["{\mathrm{roll}_B}", from=1-3, to=3-3]
  \arrow["{F(F^*(f))}", from=1-1, to=1-3]
\end{tikzcd}
~~~

```agda
      map* : ∀ {a b} → Hom a b → Hom (F* a) (F* b)
      map* f = Free.₁ f .hom

      map*-roll : ∀ {a b} (f : Hom a b) → map* f ∘ roll a ≡ roll b ∘ F.₁ (map* f)
      map*-roll f = Free.₁ f .preserves
```

The counit of our adjunction lets us extend any $F$-algebra $\alpha : \cC(F(A),A)$
to an $F$-algebra $\mathrm{fold}(\alpha) : \cC(F^{*}(A),A)$. Intuitively,
this operation lets us eliminate out of the fixpoint by describing how to
eliminate out of each layer.

```agda
      fold : ∀ {a} → Hom (F.₀ a) a → Hom (F* a) a
      fold {a} α = counit.ε (a , α) .hom

      fold-roll : ∀ {a} (α : Hom (F.₀ a) a) → fold α ∘ roll a ≡ α ∘ F.₁ (fold α)
      fold-roll {a} α = counit.ε (a , α) .preserves
```

Extending $\mathrm{roll}$ gives us the multiplication of the monad.

```agda
      mult : ∀ (x : Ob) → Hom (F* (F* x)) (F* x)
      mult x = fold (roll x)
```

We can also determine the equality of $F$-algebra morphisms $\cC(F^{*}(A),B)$
purely based off of how they behave on points.

```agda
      fold-ext
        : ∀ {a b}
        → (f g : FAlg.Hom (Free.₀ a) b)
        → (f .hom ∘ unit.η _ ≡ g .hom ∘ unit.η _)
        → f .hom ≡ g .hom
      fold-ext f g p =
        ap hom $ Equiv.injective (_ , L-adjunct-is-equiv Free⊣π) {x = f} {y = g} $
        p
```

Note that any $F$-algebra $\alpha : \cC(F(A), A)$ yields an algebra for
the algebraically free monad via extension along $\mathrm{fold}$.

```agda
    lift-alg : ∀ {a} → Hom (F.₀ a) a → Algebra-on Alg-free-monad a
    lift-alg {a = a} α .ν = fold α
    lift-alg {a = a} α .ν-unit = zag
    lift-alg {a = a} α .ν-mult =
      ap hom $ counit.is-natural (Free.₀ a) (a , α) (counit.ε (a , α))
```

Likewise, we can extract an $F$-algebra out of a monad algebra by
precomposing with $\mathrm{roll} \circ F(\eta)$: intuitively, this
lifts $F(A)$ into $F^{*}(A)$ by adding on zero extra layers, and then
passes it off to the monad algebra to be eliminated.

```agda
    lower-alg : ∀ {a} → Algebra-on Alg-free-monad a → Hom (F.₀ a) a
    lower-alg {a = a} α = α .ν ∘ roll a ∘ F.₁ (unit.η a)
```

We can also view a monad algebra $\alpha$ as a *morphism* of $F$-algebras,
as described in the following digram:

~~~{.quiver}
\begin{tikzcd}
  {F(F^*(A))} && {F(A)} \\
  \\
  {F^{*}(A)} && A
  \arrow["\alpha"', from=3-1, to=3-3]
  \arrow["{\mathrm{roll}}"', from=1-1, to=3-1]
  \arrow["{\alpha \circ \mathrm{roll}\ \circ F(\eta)}", from=1-3, to=3-3]
  \arrow["{F(\alpha)}", from=1-1, to=1-3]
\end{tikzcd}
~~~

This diagram states that we should be able to eliminate a $F(F^*(A))$
either by rolling in the $F$ and passing the resulting $F^*(A)$ off
to the monad algebra, or by eliminating the inner $F^*(A)$ first,
and *then* evaluating the remaining $F(A)$. Intuitively, this is quite
clear, but proving it involves quite a bit of algebra.

```agda
    alg*
      : ∀ {a} → (α : Algebra-on Alg-free-monad a)
      → FAlg.Hom (F* a , roll a) (a , (α .ν ∘ roll a ∘ F.₁ (unit.η a)))
    alg* {a = a} α .hom = α .ν
    alg* {a = a} α .preserves =
      α .ν ∘ roll a                                            ≡⟨ intror (F.annihilate zag) ⟩
      (α .ν ∘ roll a) ∘ (F.₁ (mult a) ∘ F.₁ (unit.η _))        ≡⟨ pull-inner (sym $ fold-roll (roll a)) ⟩
      α .ν ∘ (mult a ∘ roll (F* a)) ∘ F.₁ (unit.η _)           ≡⟨ dispersel (sym $ α .ν-mult) ⟩
      α .ν ∘ Free.₁ (α .ν) .hom ∘ roll (F* a) ∘ F.₁ (unit.η _) ≡⟨ extend-inner (map*-roll (α .ν)) ⟩
      α .ν ∘ roll a ∘ F.₁ (map* (α .ν)) ∘ F.₁ (unit.η _)       ≡⟨ centralizer (F.weave (sym (unit.is-natural _ _ _))) ⟩
      α .ν ∘ (roll a ∘ F.₁ (unit.η _)) ∘ F.₁ (α .ν)            ≡⟨ assoc _ _ _ ⟩
      (α .ν ∘ roll a ∘ F.₁ (unit.η _)) ∘ F.₁ (α .ν)            ∎
```

However, this algebra pays off, as it lets us establish an equivalence
between $F$-algebras and algebras over the algebraically free monad on $F$.

```agda
    f-algebra≃free-monad-algebra : ∀ a → Hom (F.₀ a) a ≃ Algebra-on Alg-free-monad a
    f-algebra≃free-monad-algebra a = Iso→Equiv $
      lift-alg , iso lower-alg
        (Algebra-on-pathp refl ⊙ equivl)
        equivr
      where
        equivl
          : ∀ {a} (α : Algebra-on Alg-free-monad a)
          → counit.ε (a , lower-alg α) .hom ≡ α .ν
        equivl α =
          fold-ext (counit.ε _) (alg* α) $ zag ∙ sym (α .ν-unit)

        equivr
          : ∀ {a} (α : Hom (F.₀ a) a)
          → counit.ε (a , α) .hom ∘ roll a ∘ F.₁ (unit.η _) ≡ α
        equivr {a} α =
          pulll (counit.ε (a , α) .preserves) ∙ F.cancelr zag
```

Likewise, we have an equivalence between $F$-algebra morphisms and monad
algebra morphisms.

```agda
    private module Free-EM = Cat.Reasoning (Eilenberg-Moore Alg-free-monad)

    lift-alg-hom
      : ∀ {a b} {α β}
      → FAlg.Hom (a , α) (b , β)
      → Free-EM.Hom (a , lift-alg α) (b , lift-alg β)
    lift-alg-hom f .hom = f .hom
    lift-alg-hom f .preserves =
      (sym $ ap hom $ counit.is-natural _ _ f)

    lower-alg-hom
      : ∀ {a b} {α β}
      → Free-EM.Hom (a , lift-alg α) (b , lift-alg β)
      → FAlg.Hom (a , α) (b , β)
    lower-alg-hom f .hom = f .hom
    lower-alg-hom {a} {b} {α} {β} f .preserves =
      f .hom ∘ α                                                      ≡⟨ ap₂ _∘_ refl (insertr (F.annihilate zag)) ⟩
      f .hom ∘ (α ∘ F.₁ (ε (a , α) .hom)) ∘ F.₁ (η a)                 ≡⟨ push-inner (sym (fold-roll α)) ⟩
      ⌜ f .hom ∘ ε (a , α) .hom ⌝ ∘ (roll a ∘ F.₁ (η a))              ≡⟨ ap! (f .preserves) ⟩
      (ε (b , β) .hom ∘ Free.F₁ (f .hom) .hom) ∘ (roll a ∘ F.₁ (η a)) ≡⟨ pull-inner (map*-roll (f .hom)) ⟩
      ε (b , β) .hom ∘ (roll b ∘ F.₁ (map* (f .hom))) ∘ F.₁ (η a)     ≡⟨ disperse (fold-roll β) (F.weave (sym (unit.is-natural _ _ _))) ⟩
      β ∘ F.₁ (ε (b , β) .hom) ∘ F.₁ (η b) ∘ F.₁ (f .hom)             ≡⟨ ap₂ _∘_ refl (cancell (F.annihilate zag)) ⟩
      β ∘ (F.₁ (f .hom))                                              ∎
```

Therefore, we have an [[isomorphism of precategories]] between the category
of $F$-algebras and the [[Eilenberg-Moore category]] of the monad we constructed,
giving us the appropriate universal property for an algebraically free monad.

```agda
    FAlg→Free-EM
      : Functor FAlg (Eilenberg-Moore Alg-free-monad)
    FAlg→Free-EM .F₀ (a , α) =
      a , lift-alg α
    FAlg→Free-EM .F₁ = lift-alg-hom
    FAlg→Free-EM .F-id = ext refl
    FAlg→Free-EM .F-∘ f g = ext refl

    FAlg≅Free-EM : is-precat-iso FAlg→Free-EM
    FAlg≅Free-EM .is-precat-iso.has-is-ff =
      is-iso→is-equiv $ iso lower-alg-hom
        (λ _ → trivial!)
        (λ _ → total-hom-path F-Algebras refl prop!)
    FAlg≅Free-EM .is-precat-iso.has-is-iso =
      Σ-ap-snd f-algebra≃free-monad-algebra .snd
```

### Free algebras and free relative monads

The previous construction of the algebraically free monad on $F$ only works
if we have *all* free $F$-algebras. This is a rather strong condition
on $F$: what can we do in the general setting?

First, note that the category of objects equipped with free $F$-algebras
forms a full subcategory of $\cC$.

```agda
  Free-algebras : Precategory _ _
  Free-algebras = Restrict {C = C} (Free-object (πᶠ F-Algebras))

  Free-algebras↪C : Functor Free-algebras C
  Free-algebras↪C = Forget-full-subcat
```

<!--
```agda
  private module Free-algebras = Cat.Reasoning Free-algebras
  module Free-alg (α : Free-algebras.Ob) where
    -- Rexport stuff in a more user-friendly format
    open Free-object (α .snd) public

    ob : Ob
    ob = free .fst

    alg : Hom (F.₀ ob) ob
    alg = free .snd

  open Relative-extension-system
```
-->

If we restrict along the inclusion from this category, we can construct
the free [[relative extension system]] on $F$. Unlike the algebraically free
monad on $F$, this extension system always exists, though it may be trivial
if $F$ lacks any free algebras.

```agda
  Free-relative-extension : Relative-extension-system Free-algebras↪C
  Free-relative-extension .M₀ α =
    Free-alg.ob α
  Free-relative-extension .unit {α} =
    Free-alg.unit α
  Free-relative-extension .bind {α} {β} f =
    Free-alg.fold α {Free-alg.free β} f .hom
  Free-relative-extension .bind-unit-id {α} =
    ap hom $ Free-alg.fold-unit α
  Free-relative-extension .bind-unit-∘ {α} {β} f =
    Free-alg.commute α
  Free-relative-extension .bind-∘ {α} {β} {γ} f g = ap hom $
    Free-alg.fold β f FAlg.∘ Free-alg.fold α g   ≡˘⟨ Free-alg.fold-natural α (Free-alg.fold β f) g ⟩
    Free-alg.fold α (Free-alg.fold β f .hom ∘ g) ∎
```
