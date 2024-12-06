<!--
```agda
open import Cat.Functor.Equivalence.Complete
open import Cat.Functor.Adjoint.Continuous
open import Cat.Instances.Shape.Terminal
open import Cat.Functor.Conservative
open import Cat.Functor.Equivalence
open import Cat.Diagram.Limit.Base
open import Cat.Diagram.Terminal
open import Cat.Functor.Kan.Base
open import Cat.Displayed.Total
open import Cat.Diagram.Monad
open import Cat.Prelude

import Cat.Reasoning
```
-->

```agda
module Cat.Diagram.Monad.Limits {o ℓ} {C : Precategory o ℓ} {F : Functor C C} (M : Monad-on F) where
```

<!--
```agda
private
  module EM = Cat.Reasoning (Eilenberg-Moore M)
  module C = Cat.Reasoning C
  module M = Monad-on M

open Algebra-on
open Total-hom
```
-->

# Limits in categories of algebras {defines="limits-in-categories-of-algebras"}

Suppose that $\cC$ be a category, $M$ be a [monad] on $\cC$, and
$F$ be a $\cJ$-shaped diagram of [$M$-algebras][malg] (that is, a
functor $F : \cJ \to \cC^M$ into the [Eilenberg-Moore category] of
M). Suppose that an evil wizard has given you a [limit] for the diagram
in $\cC$ which underlies $F$, but they have not (being evil and all)
told you whether $\lim F$ admits an algebra structure at all.

[monad]: Cat.Diagram.Monad.html#monads
[malg]: Cat.Diagram.Monad.html#algebras-over-a-monad
[Eilenberg-Moore category]: Cat.Diagram.Monad.html#eilenberg-moore-category
[limit]: Cat.Diagram.Limit.Base.html

Perhaps we can make this situation slightly more concrete, by working in
a category _equivalent to_ an Eilenberg-Moore category: If we have two
groups $G$, $H$, considered as a discrete diagram, then the limit our
evil wizard would give us is the product $U(G) \times U(H)$ in $\Sets$.
But we [already know] we can equip this product with a "pointwise" group
structure! Does this result generalise?

[already know]: Algebra.Group.Cat.FinitelyComplete.html#direct-products

The answer is positive, though this is one of those cases where abstract
nonsense can not help us: gotta roll up our sleeves and calculate away.
Suppose we have a diagram as in the setup --- we'll show that the
functor $U : \cC^M \to \cC$ both preserves and _reflects_ limits,
in that $K$ is a limiting cone if, and only if, $U(K)$ is.

<!--
```agda
module _ {jo jℓ} {J : Precategory jo jℓ} (F : Functor J (Eilenberg-Moore M)) where
  private
    module J = Precategory J
    module F = Functor F
    module FAlg j = Algebra-on (F.₀ j .snd)
  open Functor
  open _=>_
```
-->

That $U$ preserves limits follows immediately from the fact that it is a
right adjoint: the non-trivial part is showing that it reflects them.

```agda
  Forget-EM-preserves-limits : preserves-limit Forget-EM F
  Forget-EM-preserves-limits = right-adjoint-is-continuous Free-EM⊣Forget-EM
```

We begin with the following key lemma: Write $K : \cC$ for the limit of
a diagram $\cJ \xto{F} \cC^M \xto{U} \cC$. If $K$ carries an $M$-algebra
structure $\nu$, and the limit projections $\psi : K \to F(j)$ are
$M$-algebra morphisms, then $(K, \nu)$ is the limit of $F$ in $\cC^M$.

```agda
  make-algebra-limit
    : ∀ {K : Functor ⊤Cat C} {eps : K F∘ !F => Forget-EM F∘ F}
    → (lim : is-ran !F (Forget-EM F∘ F) K eps)
    → (nu : Algebra-on M (K .F₀ tt))
    → (∀ j → is-limit.ψ lim j C.∘ nu .ν ≡ FAlg.ν j C.∘ M.M₁ (is-limit.ψ lim j))
    → make-is-limit F (K .F₀ tt , nu)
  make-algebra-limit lim apex-alg comm = em-lim where
    module lim = is-limit lim
    open make-is-limit
    module apex = Algebra-on apex-alg
    open _=>_
```

The assumptions to this lemma are actually rather hefty: they pretty
much directly say that $K$ was already the limit of $F$. However, with
this more "elementary" rephrasing, we gain a bit of extra control which
will be necessary for _constructing_ limits in $\cC^M$ out of limits in
$\cC$ later.

```agda
    em-lim : make-is-limit F _
    em-lim .ψ j .hom = lim.ψ j
    em-lim .ψ j .preserves = comm j
    em-lim .commutes f    = ext (lim.commutes f)
    em-lim .universal eps p .hom =
      lim.universal (λ j → eps j .hom) (λ f i → p f i .hom)
    em-lim .factors eps p =
      ext (lim.factors _ _)
    em-lim .unique eps p other q =
      ext (lim.unique _ _ _ λ j i → q j i .hom)
    em-lim .universal eps p .preserves = lim.unique₂ _
      (λ f → C.pulll (F.F₁ f .preserves)
           ∙ C.pullr (sym (M.M-∘ _ _) ∙ ap M.M₁ (ap hom (p f))))
      (λ j → C.pulll (lim.factors _ _)
           ∙ eps j .preserves)
      (λ j → C.pulll (comm j)
           ∙ C.pullr (sym (M.M-∘ _ _) ∙ ap M.M₁ (lim.factors _ _)))
```

This key lemma also doubles as a proof that the underlying object
functor $U$ reflects limits: We already had an algebra structure
"upstairs"!

```agda
  Forget-reflects-limits : reflects-limit Forget-EM F
  Forget-reflects-limits {K} {eps} lim = to-is-limitp
    (make-algebra-limit lim (K .F₀ tt .snd) (λ j → eps .η j .preserves))
    trivial!
```

Having shown that $U$ reflects the property of _being a limit_, we now
turn to showing it reflects limits in general. Using our key lemma, it
will suffice to construct an algebra structure on $\lim_j UF(j)$, then
show that the projection maps $\psi_j : (\lim_j UF(j)) \to UF(j)$ extend
to algebra homomorphisms.

```agda
  Forget-lift-limit : Limit (Forget-EM F∘ F) → Limit F
  Forget-lift-limit lim-over = to-limit $ to-is-limit $ make-algebra-limit
    (Limit.has-limit lim-over) apex-algebra (λ j → lim-over.factors _ _)
    where
    module lim-over = Limit lim-over
    module lim = Ran lim-over
```

An algebra structure consists, centrally, of a map $M(\lim_j UF(j)) \to
\lim_j UF(j)$: a map _into_ the limit. Maps into limits are the happy
case, since $\lim_j UF(j)$ is essentially defined by a (natural)
isomorphism between the sets $\hom(X, \lim_j UF(j))$ and $\lim_j \hom(X,
UF(j))$, the latter limit being computed in $\Sets$. We understand
limits of sets very well: they're (subsets of) sets of tuples!

Our algebra map is thus defined _componentwise_: Since we have a bunch
of maps $\nu_j : M(UF(j)) \to UF(j)$, coming from the algebra structures
on each $F(j)$, we can "tuple" them into a big map $\nu = \langle \nu_j
\psi_j \rangle _j$. Abusing notation slightly, we're defining $\nu(a, b,
...)$ to be $(\nu_a(a), \nu_b(b), ...)$.

```agda
    apex-algebra : Algebra-on M lim-over.apex
    apex-algebra .ν =
      lim-over.universal (λ j → FAlg.ν j C.∘ M.M₁ (lim-over.ψ j)) comm where abstract
      comm : ∀ {x y} (f : J.Hom x y)
            → F.₁ f .hom C.∘ FAlg.ν x C.∘ M.M₁ (lim-over.ψ x)
            ≡ FAlg.ν y C.∘ M.M₁ (lim-over.ψ y)
      comm {x} {y} f =
        F.₁ f .hom C.∘ FAlg.ν x C.∘ M.M₁ (lim-over.ψ x)        ≡⟨ C.extendl (F.₁ f .preserves) ⟩
        FAlg.ν y C.∘ M.M₁ (F.₁ f .hom) C.∘ M.M₁ (lim-over.ψ x) ≡˘⟨ C.refl⟩∘⟨ M.M-∘ _ _ ⟩
        FAlg.ν y C.∘ M.M₁ (F.₁ f .hom C.∘ lim-over.ψ x)        ≡⟨ C.refl⟩∘⟨ ap M.M₁ (lim-over.commutes f) ⟩
        FAlg.ν y C.∘ M.M₁ (lim-over.ψ y)                            ∎
```

To show that $\nu$ really is an algebra structure, we'll reason
componentwise, too: we must show that $\nu(\eta_a, \eta_b, ...)$ is
the identity map: but we can compute

$$
\nu(\eta_a, \eta_b, ...) = (\nu_a\eta_a, \nu_b\eta_b, ...) = (\id, \id, ...) = \id
$$!

<details>
<summary>
The other condition, compatibility with $M$'s multiplication, is
analogous. Formally, the computation is a bit less pretty, but it's no
more complicated.
</summary>

```agda
    apex-algebra .ν-unit = lim-over.unique₂ _ lim-over.commutes
      (λ j → C.pulll (lim-over.factors _ _)
          ·· C.pullr (sym $ M.unit.is-natural _ _ _)
          ·· C.cancell (FAlg.ν-unit j))
      (λ j → C.idr _)
    apex-algebra .ν-mult = lim-over.unique₂ _
      (λ f → C.pulll $ C.pulll (F.₁ f .preserves)
           ∙ C.pullr (sym (M.M-∘ _ _) ∙ ap M.M₁ (lim-over.commutes f)))
      (λ j → C.pulll (lim-over.factors _ _)
          ·· C.pullr (sym (M.M-∘ _ _) ∙ ap M.M₁ (lim-over.factors _ _) ∙ M.M-∘ _ _)
          ·· C.extendl (FAlg.ν-mult j)
          ·· ap (FAlg.ν j C.∘_) (M.mult.is-natural _ _ _)
          ·· C.assoc _ _ _)
      (λ j → C.pulll (lim-over.factors _ _))
```

</details>

Putting our previous results together, we conclude by observing that, if
$\cC$ is a complete category, then so is $\cC^M$, regardless of how
ill-behaved the monad $M$ might be.

```agda
Eilenberg-Moore-is-complete
  : ∀ {a b} → is-complete a b C → is-complete a b (Eilenberg-Moore M)
Eilenberg-Moore-is-complete complete F =
  Forget-lift-limit F (complete _)
```
