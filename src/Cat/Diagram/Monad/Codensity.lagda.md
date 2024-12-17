<!--
```agda
open import Cat.Instances.Functor
open import Cat.Functor.Kan.Base
open import Cat.Diagram.Monad
open import Cat.Prelude

import Cat.Functor.Reasoning as Func
import Cat.Reasoning as Cat
```
-->

```agda
module Cat.Diagram.Monad.Codensity where
```

<!--
```agda
private variable
  o ℓ : Level
  A B : Precategory o ℓ
open Monad-on
open _=>_
```
-->

# Codensity monads

Let $F : \cA \to \cB$ be a functor with a [[left adjoint]] $G :
\cB \to \cA$. Some pretty [standard abstract nonsense][adjm] tells
us that the composite $GF$ is a [monad] in $\cB$, with the unit and
multiplication inherited from the adjunction $G \dashv F$. The easiest
cases to picture are when $\cB$ is $\Sets_\kappa$, and $F$ is the
"underlying set" functor from an algebraic category (like [[monoids|free
monoid]] or [[groups|free group]]). What's slightly more interesting is
that functors _without_ left adjoints may also induce monads!

[adjm]: Cat.Functor.Adjoint.Monad.html
[monad]: Cat.Diagram.Monad.html

This is called the **codensity monad** of the functor $F$, and it exists
whenever $\cB$ admits [[limits]] indexed by categories the [size] of
$\cA$. When $F$ does have a left adjoint, its codensity monad
coincides with the ordinary monad-from-an-adjunction construction.
Rather than unfolding the necessary limits here, though, we defer to
general categorical machinery: [[right Kan extensions]].

[size]: 1Lab.intro.html#universes-and-size-issues

The really, really short of it is that the codensity monad of $F$ is the
right Kan extension of $F$ along itself, $\Ran_F F$.

<!--
```agda
module _ (F : Functor A B) (R : Ran F F) where
  open Ran R
  private
    module A = Cat A
    module B = Cat B
    module F = Func F
```
-->

Constructing the monad structure on the $\Ran_F F$ functor is a bit
involved, but it does serve as a very illustrative application of the
universal property of right Kan extensions. Recall that, by definition,
if we have a natural transformation $GF \To F$ (for our choice of
functor $G$), then this extends to a (unique) natural transformation $G
\To \Ran_F F$.

For example, the unit map $\eta$ is obtained by extending the identity
natural transformation $\id : \Id F \To F$, which is implicit witnessing
commutativity of the $F$ -- $\Id$ -- $F$ triangle below.

~~~{.quiver}
\[\begin{tikzcd}
  {\mathcal{A}} && B \\
  \\
  B
  \arrow[""{name=0, anchor=center, inner sep=0}, "{\mathrm{Ran}_F F}"{description, pos=0.7}, curve={height=-12pt}, dashed, from=3-1, to=1-3]
  \arrow[""{name=1, anchor=center, inner sep=0}, "{\mathrm{Id}}"{description, pos=0.3}, curve={height=12pt}, from=3-1, to=1-3]
  \arrow["F", from=1-1, to=1-3]
  \arrow["F"', from=1-1, to=3-1]
  \arrow["\eta"', shorten <=4pt, shorten >=4pt, Rightarrow, from=1, to=0]
\end{tikzcd}\]
~~~

```agda
    unit-nt : Id F∘ F => F
    unit-nt .η x = B.id
    unit-nt .is-natural x y f = B.id-comm-sym
```

For the multiplication map, observe that we can piece together a natural
transformation

$$
(\Lan_F(F))^2Fx \to \Lan_F(F)Fx \to Fx
$$,

using the canonical natural transformation $\eps : \Lan_F(F)F \To F$.
Extending _this_ map, then, gives us a natural transformation
$(\Lan_F(F))^2 \To \Lan_F(F)$.

```agda
    mult-nt : (Ext F∘ Ext) F∘ F => F
    mult-nt .η x = eps .η x B.∘ Ext.₁ (eps .η x)
    mult-nt .is-natural x y f =
      (eps .η y B.∘ Ext.₁ (eps .η y)) B.∘ Ext.₁ (Ext.₁ (F.₁ f)) ≡⟨ Ext.extendr (eps .is-natural _ _ _) ⟩
      (eps .η y B.∘ Ext.F₁ (F.₁ f)) B.∘ Ext.₁ (eps .η x)        ≡⟨ B.pushl (eps .is-natural _ _ _) ⟩
      F.₁ f B.∘ eps .η x B.∘ Ext.₁ (eps .η x)                   ∎

  Codensity : Monad-on Ext
  Codensity .unit = σ unit-nt
  Codensity .mult = σ mult-nt
```

<details>
<summary>Proving that these two extended natural transformations really
do comprise a monad structure is a routine application of the uniqueness
properties of right Kan extensions. The real noise comes from having to
construct auxiliary natural transformations representing each pair of
maps we want to compute with.</summary>

```agda
  Codensity .μ-unitr {x = x} = path ηₚ x where
    nat₁ : Ext => Ext
    nat₁ .η x = σ mult-nt .η x B.∘ Ext.₁ (σ unit-nt .η x)
    nat₁ .is-natural x y f = Ext.extendr (σ unit-nt .is-natural x y f)
                           ∙ B.pushl (σ mult-nt .is-natural _ _ _)

    abstract
      path : nat₁ ≡ idnt
      path = σ-uniq₂ eps
        (ext λ x → sym
          ( B.pulll (σ-comm ηₚ x)
          ∙ Ext.cancelr (σ-comm ηₚ x)))
        (ext λ _ → B.intror refl)

  Codensity .μ-unitl {x = x} = path ηₚ x where
    nat₁ : Ext => Ext
    nat₁ .η x = σ mult-nt .η x B.∘ σ unit-nt .η (Ext.₀ x)
    nat₁ .is-natural x y f = B.extendr (σ unit-nt .is-natural _ _ _)
                           ∙ B.pushl (σ mult-nt .is-natural _ _ _)

    abstract
      path : nat₁ ≡ idnt
      path = σ-uniq₂ eps
        (ext λ x → sym $
            B.pulll (σ-comm ηₚ x)
          ·· B.pullr (sym (σ unit-nt .is-natural _ _ _))
          ·· B.cancell (σ-comm ηₚ x))
        (ext λ _ → B.intror refl)

  Codensity .μ-assoc {x = x} = path ηₚ x where
    mult' : (Ext F∘ Ext F∘ Ext) F∘ F => F
    mult' .η x = eps .η x B.∘ Ext.₁ (mult-nt .η x)
    mult' .is-natural x y f = Ext.extendr (mult-nt .is-natural _ _ _)
                            ∙ B.pushl (eps .is-natural _ _ _)

    sig₁ : Ext F∘ Ext F∘ Ext => Ext
    sig₁ .η x = σ mult-nt .η x B.∘ Ext.₁ (σ mult-nt .η x)
    sig₁ .is-natural x y f = Ext.extendr (σ mult-nt .is-natural _ _ _)
                            ∙ B.pushl (σ mult-nt .is-natural _ _ _)

    sig₂ : Ext F∘ Ext F∘ Ext => Ext
    sig₂ .η x = σ mult-nt .η x B.∘ σ mult-nt .η (Ext.₀ x)
    sig₂ .is-natural x y f = B.extendr (σ mult-nt .is-natural _ _ _)
                            ∙ B.pushl (σ mult-nt .is-natural _ _ _)

    abstract
      path : sig₁ ≡ sig₂
      path = σ-uniq₂ {M = Ext F∘ Ext F∘ Ext} mult'
        (ext λ x → sym $
            B.pulll (σ-comm ηₚ x)
          ∙ Ext.pullr (σ-comm ηₚ x))
        (ext λ x → sym $
             B.pulll (σ-comm ηₚ x)
          ·· B.pullr (sym (σ mult-nt .is-natural _ _ _))
          ·· B.pulll (σ-comm ηₚ x)
           ∙ Ext.pullr refl)
```
</details>

To understand what the codensity monad _represents_, recall that
adjoints can be understood as [[efficient solutions|universal-morphism]] to "optimisation
problems". But when a functor does _not_ admit a left adjoint, we
conclude that there is no most efficient solution; This doesn't mean
that we can't _approximate_ a solution, though! And indeed, this kind of
approximation is exactly what right Kan extensions are for.
