<!--
```agda
open import Cat.Diagram.Product.Solver
open import Cat.Internal.Opposite
open import Cat.Diagram.Pullback
open import Cat.Diagram.Terminal
open import Cat.Diagram.Product
open import Cat.Prelude

import Cat.Internal.Reasoning
import Cat.Internal.Base
import Cat.Reasoning
```
-->

```agda
module Cat.Internal.Functor.Outer
  {o ℓ} {C : Precategory o ℓ}
  where
```

<!--
```agda
open import Cat.Reasoning C
open import Cat.Internal.Base C
open Internal-hom
```
-->

# Outer functors

The category $\Sets$ is not [[strict|strict category]], so it is not
[internal] to any category of sets, even setting aside size issues.
However, $\Sets$ still plays an important role in (strict) category
theory, by passing to the co[presheaf] categories $\cC \to \Sets$.

[internal]: Cat.Internal.Base.html
[presheaf]: Cat.Functor.Base.html#presheaf-precategories

We wish to relativize this to an arbitrary base category $\cC$, not just
$\thecat{Sets}$. Specifically, if $\bC$ is a category internal to $\cC$,
we want to define "functor from $\bC \to \cC$" --- which we call an
**outer functor**[^1]. We will employ the fibration/family
correspondence: a family of sets $B \to \Sets$ is the same as a function
$E \to B$ --- and we know how to relativise functions!

[^1]: The terminology here is somewhat inconsistent. [@Borceux:vol1]
calls these functors _internal base-valued functors_, while in [@SIGL]
they are referred to as _category actions_.

```agda
module _ (ℂ : Internal-cat) where
  open Cat.Internal.Reasoning ℂ

  record Outer-functor : Type (o ⊔ ℓ) where
    no-eta-equality
```

The first piece of data we require is some object $P : \cC$ that will
act like the disjoint union of the image of our functor. If the base
category were $\Sets$, this would be the type $\Sigma_(x : \ica{C})
P(x)$. The next piece of data, $P_0$, corresponds to the first
projection morphism: it assigns (generalised) objects of $P$ to objects
of $\bC$.

```agda
    field
      ∫P : Ob
      P₀ : ∀ {Γ} → Hom Γ ∫P → Hom Γ C₀
```

The next data captures how $\bC$'s morphisms act on the fibres. Since
the family-fibration correspondence takes dependency to sectioning, we
require an assigment sending maps $f : P_0(x) \to y$ to maps $P_1(f) :
\Gamma \to P$, which satisfy $y = P_0(P_1(f))$.

```agda
      P₁ : ∀ {Γ} (px : Hom Γ ∫P) {y : Hom Γ C₀} → Homi (P₀ px) y → Hom Γ ∫P
      P₁-tgt : ∀ {Γ} (px : Hom Γ ∫P) {y : Hom Γ C₀}
             → (f : Homi (P₀ px) y) → y ≡ P₀ (P₁ px f)
```

Next, we have the functoriality conditions, which are straightforward
(modulo indexing).

```agda
      P-id : ∀ {Γ} (px : Hom Γ ∫P) → P₁ px (idi (P₀ px)) ≡ px
      P-∘ : ∀ {Γ} (px : Hom Γ ∫P) {y z : Hom Γ C₀} (f : Homi y z) (g : Homi (P₀ px) y)
          → P₁ px (f ∘i g) ≡ P₁ (P₁ px g) (adjusti (P₁-tgt px g) refl f)
```

Finally, the naturality conditions that allow us to work using
generalized objects.

```agda
      P₀-nat : ∀ {Γ Δ} → (px : Hom Δ ∫P) → (σ : Hom Γ Δ) → P₀ px ∘ σ ≡ P₀ (px ∘ σ)
      P₁-nat : ∀ {Γ Δ} (px : Hom Δ ∫P) {y : Hom Δ C₀} (f : Homi (P₀ px) y) (σ : Hom Γ Δ)
        → P₁ px f ∘ σ ≡ P₁ (px ∘ σ) (adjusti (P₀-nat px σ) refl (f [ σ ]))

open Outer-functor
```

Another perspective on outer functors is that they are the _internal
discrete opfibrations_ over $\ica{C}$. The object $P$ is the [[total
category]], the map $P_0$ is the fibration itself, and $P_1$ captures the
lifting into opcartesian maps.

<!-- [TODO: Reed M, 28/04/2023]
Link to the page on discrete opfibrations when it is written!
-->

We can obtain internal [discrete fibrations] by looking at outer
functors from the [internal opposite category] of $\ica{C}$.

[discrete fibrations]: Cat.Displayed.Cartesian.Discrete.html
[internal opposite category]: Cat.Internal.Opposite.html

## Internal Hom functors

The canonical example of an outer functor is the internal analog of the
hom functor $\cC(X,-)$. We require the following data: $\cC$ is
[[finitely complete]], $\bC$ is a category internal to $\cC$, and $x :
\cC(\top, \bC_0)$ is a _global_ object of $\bC$ --- an object of $\bC$
in the empty context.

```agda
module _ (pb : has-pullbacks C) (term : Terminal C) (ℂ : Internal-cat) where
  open Cat.Internal.Reasoning ℂ
  open Pullbacks C pb
  open Terminal term

  Internal-hom-from : (x : Hom top C₀) → Outer-functor ℂ
  Internal-hom-from x = outf where
```

Recall that defining an outer functor on $\ica{C}$ requires choosing an
object $P : \cC$ to play the role of total space. For the outer functor
corresponding to a Hom-functor, this ought to be the object of morphisms
with domain $x$. We can encode this internally with the following
pullback:

~~~{.quiver}
\begin{tikzcd}
  {C_x} && \top \\
  \\
  {C_1} && {C_0}
  \arrow["src", from=3-1, to=3-3]
  \arrow["x", from=1-3, to=3-3]
  \arrow[from=1-1, to=3-1]
  \arrow[from=1-1, to=1-3]
  \arrow["\lrcorner"{anchor=center, pos=0.125}, draw=none, from=1-1, to=3-3]
\end{tikzcd}
~~~

The projection from the total space to $C_0$ takes a generalized element
of $C_x$ to its codomain. The lifting morphism $P_1$ follows from the
universal property of the pullback.

```agda
    open Pullback (pb src x) renaming (apex to Cₓ)

    outf : Outer-functor ℂ
    outf .∫P = Cₓ
    outf .P₀ fₓ = tgt ∘ p₁ ∘ fₓ
    outf .P₁ fₓ {y = y} g = universal coh
      module hom-from-action where abstract
        coh : src ∘ (g ∘i homi (p₁ ∘ fₓ)) .ihom ≡ x ∘ !
        coh =
          src ∘ (g ∘i homi (p₁ ∘ fₓ)) .ihom ≡⟨ (g ∘i homi (p₁ ∘ fₓ)) .has-src ⟩
          src ∘ p₁ ∘ fₓ ≡⟨ pulll square ⟩
          (x ∘ p₂) ∘ fₓ ≡⟨ pullr (sym (!-unique _)) ⟩
          x ∘ ! ∎
```

<details>
<summary>The functoriality constraint is analogous to that for the
ordinary $\hom$-functors, though here it is obscured by all the
pullbacks. The naturality constraints similarly follow from uniqueness
of maps into a limit.
</summary>

```agda
    outf .P₁-tgt fₓ {y = y} g = tgt-coh where abstract
      tgt-coh : y ≡ tgt ∘ p₁ ∘ universal (hom-from-action.coh fₓ g)
      tgt-coh =
        y                                 ≡˘⟨ (g ∘i homi (p₁ ∘ fₓ)) .has-tgt ⟩
        tgt ∘ (g ∘i homi (p₁ ∘ fₓ)) .ihom ≡˘⟨ ap (tgt ∘_) p₁∘universal ⟩
        tgt ∘ p₁ ∘ universal _            ∎
    outf .P-id fₓ =
      sym $ unique (sym (ap ihom (idli _))) (sym (!-unique _))
    outf .P-∘ fₓ g h = unique
      (p₁∘universal
      ∙ ap ihom (sym $ associ _ _ _)
      ∙ ∘i-ihom
          (sym (ap (src ∘_) p₁∘universal ∙ (h ∘i homi (p₁ ∘ fₓ)) .has-src))
          (sym (ap (tgt ∘_) p₁∘universal ∙ (h ∘i homi (p₁ ∘ fₓ)) .has-tgt))
          refl refl (sym p₁∘universal))
      p₂∘universal
    outf .P₀-nat fₓ σ = sym (assoc _ _ _) ∙ ap (tgt ∘_) (sym (assoc _ _ _))
    outf .P₁-nat fₓ g σ = unique
      (pulll p₁∘universal
        ∙ ap ihom (∘i-nat g (homi (p₁ ∘ fₓ)) σ)
        ∙ ∘i-ihom
            (sym (assoc _ _ _) ∙ ap (src ∘_) (sym (assoc _ _ _)))
            (sym (assoc _ _ _) ∙ ap (tgt ∘_) (sym (assoc _ _ _)))
            refl refl (sym (assoc _ _ _)))
      (sym (!-unique _))
```
</details>

The _contravariant_ internal $\hom$ functor is defined by duality, which
carries "pullback along $\src$" to "pullback along $\tgt$.".
This outer functor plays the role of the Yoneda embedding.

```agda
  Internal-hom-into : (x : Hom top C₀) → Outer-functor (ℂ ^opi)
  Internal-hom-into x = outf where
    open Pullback (pb tgt x) renaming (apex to Cₓ)
```

<details>
<summary>We omit this construction due to its similarity with the
covariant construction, performed above.
</summary>

```agda
    outf : Outer-functor (ℂ ^opi)
    outf .∫P = Cₓ
    outf .P₀ fₓ = src ∘ p₁ ∘ fₓ
    outf .P₁ fₓ g = universal coh
      module hom-into-action where abstract
        coh : tgt ∘ (homi (p₁ ∘ fₓ) ∘i op-ihom g) .ihom ≡ x ∘ !
        coh =
          tgt ∘ (homi (p₁ ∘ fₓ) ∘i op-ihom g) .ihom ≡⟨ (homi (p₁ ∘ fₓ) ∘i op-ihom g) .has-tgt ⟩
          tgt ∘ p₁ ∘ fₓ ≡⟨ pulll square ⟩
          (x ∘ p₂) ∘ fₓ ≡⟨ pullr (sym (!-unique _)) ⟩
          x ∘ ! ∎
    outf .P₁-tgt fₓ {y} g = src-coh where abstract
      src-coh : y ≡ src ∘ p₁ ∘ universal (hom-into-action.coh fₓ g)
      src-coh =
        sym (ap (src ∘_) p₁∘universal
        ∙ (homi (p₁ ∘ fₓ) ∘i op-ihom g) .has-src)
    outf .P-id fₓ =
      sym $ unique (sym (ap ihom (idri _))) (sym (!-unique _))
    outf .P-∘ fₓ g h =
      unique
        (p₁∘universal
         ∙ ap ihom (associ _ _ _)
         ∙ ∘i-ihom refl
             (sym (ap (src ∘_) p₁∘universal ∙ (homi (p₁ ∘ fₓ) ∘i op-ihom h) .has-src))
             (sym (ap (tgt ∘_) p₁∘universal ∙ (homi (p₁ ∘ fₓ) ∘i op-ihom h) .has-tgt))
             (sym p₁∘universal) refl)
        p₂∘universal
    outf .P₀-nat fₓ σ =
      sym (assoc _ _ _)
      ∙ ap (src ∘_) (sym (assoc _ _ _))
    outf .P₁-nat fₓ g σ =
      unique
        (pulll p₁∘universal
        ∙ ap ihom (∘i-nat _ _ _)
        ∙ ∘i-ihom refl
             (sym (assoc _ _ _) ∙ ap (src ∘_) (sym (assoc _ _ _)))
             (sym (assoc _ _ _) ∙ ap (tgt ∘_) (sym (assoc _ _ _)))
             (sym (assoc _ _ _)) refl)
        (sym (!-unique _))
```
</details>

## Outer natural transformations

If $\cC$ is a category, $\bC$ is internal to $\cC$, and $P, Q$ are two
outer functors on $\bC$, we can define the **outer natural
transformations** $P \to Q$: They are (mostly) given by transformations
$\cC(-, \int P)$ to $\cC(-, \int Q)$, together with a bit of naturality
data.

```agda
module _ {ℂ : Internal-cat} where
  open Internal-cat ℂ
  record _=>o_ (P Q : Outer-functor ℂ) : Type (o ⊔ ℓ) where
    no-eta-equality
    field
      ηo : ∀ {Γ} → Hom Γ (P .∫P) → Hom Γ (Q .∫P)
```

The first condition corresponds to indexing: Outer natural
transformations map elements of $\int P$ over $x$ to elements of $\int
Q$ over $x$.

```agda
      ηo-fib : ∀ {Γ} (px : Hom Γ (P .∫P)) → Q .P₀ (ηo px) ≡ P .P₀ px
```

Over this coherence, we can state the naturality condition. It should be
familiar, since, modulo the aforementioned coherence, it says
$\eta P_1(f) = Q_1(f\eta)$.

```agda
      is-naturalo : ∀ {Γ : Ob} (px : Hom Γ (P .∫P)) (y : Hom Γ C₀)
        → (f : Homi (P .P₀ px) y)
        → ηo (P .P₁ px f) ≡ Q .P₁ (ηo px) (adjusti (sym (ηo-fib px)) refl f)
```

The final naturality condition is stability under substitution, allowing
us to work in the internal language of $\cC$.

```agda
      ηo-nat : ∀ {Γ Δ} (px : Hom Δ (P .∫P)) (σ : Hom Γ Δ) → ηo px ∘ σ ≡ ηo (px ∘ σ)

  open _=>o_
```

<!--
```agda
  Outer-nat-path
    : ∀ {F G : Outer-functor ℂ} {α β : F =>o G}
    → (∀ {Γ} (px : Hom Γ (F .∫P)) → α .ηo px ≡ β .ηo px)
    → α ≡ β
  Outer-nat-path p i .ηo px = p px i
  Outer-nat-path {G = G} {α = α} {β = β} p i .ηo-fib px =
    is-prop→pathp (λ i → Hom-set _ _ (G .P₀ (p px i)) _)
      (α .ηo-fib px)
      (β .ηo-fib px) i
  Outer-nat-path {F = F} {G = G} {α = α} {β = β} p i .is-naturalo px y f j =
    is-set→squarep (λ i j → Hom-set _ _)
      (p (F .P₁ px f))
      (α .is-naturalo px y f)
      (β .is-naturalo px y f)
      (λ i → G .P₁ (p px i)
        (adjusti (sym (Outer-nat-path {α = α} {β = β} p i .ηo-fib px)) refl f))
      i j
  Outer-nat-path {α = α} {β = β} p i .ηo-nat px σ =
    is-prop→pathp (λ i → Hom-set _ _ (p px i ∘ σ) (p (px ∘ σ) i))
      (α .ηo-nat px σ)
      (β .ηo-nat px σ) i

unquoteDecl H-Level-=>o = declare-record-hlevel 2 H-Level-=>o (quote _=>o_)
```
-->

## Some outer functors

One important outer functor is the **constant outer functor** on an
object $X : \cC$, which can be constructed if $\cC$ has products. This
is the internalized version of the [chaotic bifibration].

[chaotic bifibration]: Cat.Displayed.Instances.Chaotic.html

<!--
```agda
module _ (prods : has-products C) {ℂ : Internal-cat} where
  open Binary-products C prods
  open Internal-cat ℂ
  open _=>o_
```
-->

The total space of this functor is the product $X \times \bC_0$, and the
projection map is simply the second projection.

```agda
  ConstO : (X : Ob) → Outer-functor ℂ
  ConstO X .∫P = X ⊗₀ C₀
  ConstO X .P₀ f = π₂ ∘ f
```

Lifting is given by the universal map into the product.

```agda
  ConstO X .P₁ px {y} f = ⟨ π₁ ∘ px , y ⟩
  ConstO X .P₁-tgt px {y = y} f = sym π₂∘⟨⟩
```


For once, the naturality constraints are not egregious: In fact, since
they are all facts about products, they can all be solved automatically.

```agda
  ConstO X .P-id px = products! prods
  ConstO X .P-∘ px f g = products! prods
  ConstO X .P₀-nat px σ = sym (assoc _ _ _)
  ConstO X .P₁-nat px f σ = products! prods
```
</details>

If $x \to y$ is a map in $\cC$, then it defines an outer natural
transformation between the associated constant outer functors. Here too
we can apply automation to satisfy the coherence constraints.

```agda
  const-nato : ∀ {X Y : Ob} → Hom X Y → ConstO X =>o ConstO Y
  const-nato f .ηo g = ⟨ f ∘ π₁ ∘ g , π₂ ∘ g ⟩
  const-nato f .ηo-fib px          = products! prods
  const-nato f .is-naturalo px y g = products! prods
  const-nato f .ηo-nat px σ        = products! prods
```
