<!--
```agda
open import Cat.Diagram.Limit.Finite
open import Cat.Instances.Coalgebras
open import Cat.Diagram.Equaliser
open import Cat.Diagram.Pullback
open import Cat.Diagram.Terminal
open import Cat.Diagram.Comonad
open import Cat.Diagram.Product
open import Cat.Displayed.Total
open import Cat.Functor.Adjoint
open import Cat.Prelude

import Cat.Functor.Reasoning as Func
import Cat.Reasoning
```
-->

```agda
module Cat.Instances.Coalgebras.Cartesian
  {o ℓ} {C : Precategory o ℓ} {F : Functor C C} (W : Comonad-on F)
  (fc : Finitely-complete C) (lex : is-lex F)
  where
```

<!--
```agda
open Finitely-complete fc
open Cat.Reasoning C
open is-lex lex

open Total-hom

open Coalgebra-on
open Comonad-on W

open is-pullback
open Pullback

private module W = Func F
```
-->

# Finite limits in categories of coalgebras {defines="finite-limits-of-coalgebras"}

If $(W, \eps, \delta)$ is an arbitrary [[comonad]] on a category $\cC$,
there's not much we can say about general limits in the [[category of
$W$-coalgebras|category of coalgebras]]. Indeed, precisely dualising the
case for [[categories of algebras over a monad|eilenberg-moore
category]], where limits are inherited from the base category but
colimits are onerous, for $\cC_W$, colimits come easily but _limits_ are
onerous. However, if $W$ preserves a certain class of limits, then the
forgetful functor $U : \cC_W \to \cC$ preserves _and_ reflects those
same limits.

<!--
```agda
module
  _ {(A , α') (B , β') (X , ξ') : Coalgebras.Ob W}
    (f : Coalgebras.Hom W (A , α') (X , ξ'))
    (g : Coalgebras.Hom W (B , β') (X , ξ'))
  where
  private
    module α = Coalgebra-on α'
    module β = Coalgebra-on β'
    module ξ = Coalgebra-on ξ'
    open α renaming (ρ to α) using ()
    open β renaming (ρ to β) using ()
    open ξ renaming (ρ to ξ) using ()
```
-->

In this module, we're interested in the specific case where $W$ is a
[[left exact functor]], i.e., it preserves [[finite limits]].
Accordingly, $\cC_W$ will have all finite limits, and they will be
computed as in $\cC$. The more general computation can be found in
[`Cat.Instances.Coalgebras.Limits`](Cat.Instances.Coalgebras.Limits.html).

We start by showing that $U$ _reflects_ [[pullbacks]], since that's what
we'll use to construct pullbacks in $\cC_W$.

```agda
  is-pullback-coalgebra
    : ∀ {(P , π) : Coalgebras.Ob W}
        {π₁ : Coalgebras.Hom W (P , π) (A , α')}
        {π₂ : Coalgebras.Hom W (P , π) (B , β')}
    → is-pullback C (π₁ .hom) (f .hom) (π₂ .hom) (g .hom)
    → is-pullback (Coalgebras W) π₁ f π₂ g
  is-pullback-coalgebra {P , π'} {π₁} {π₂} pb = pb' where
```

To be precise, we're showing that, if a square

~~~{.quiver .short-05}
\[\begin{tikzcd}[ampersand replacement=\&]
  {(P,\pi)} \&\& {(B,\beta)} \\
  \\
  {(A,\alpha)} \&\& {(X,\xi)}
  \arrow["{\pi_2}", from=1-1, to=1-3]
  \arrow["g", from=1-3, to=3-3]
  \arrow["{\pi_1}"', from=1-1, to=3-1]
  \arrow["f"', from=3-1, to=3-3]
\end{tikzcd}\]
~~~

is sent by $U$ to a pullback in $\cC$, it must already have been a
pullback in $\cC_W$.

<!--
```agda
    module π = Coalgebra-on π'
    open π renaming (ρ to π) using ()
    pres = pres-pullback pb
    module p  = is-pullback pb
    module p' = is-pullback pres

    module
      _ {(Q , χ') : Coalgebras.Ob W}
        (p1 : Coalgebras.Hom W (Q , χ') (A , α'))
        (p2 : Coalgebras.Hom W (Q , χ') (B , β'))
        (wit : f .hom ∘ p1 .hom ≡ g .hom ∘ p2 .hom)
      where
      module χ = Coalgebra-on χ'
      open χ renaming (ρ to χ) using ()
```
-->

Since $W$ preserves pullbacks, $WP$ is a pullback as well. If we are
given a coalgebra $(Q, \chi)$ together with a span of homomorphisms

$$
(A, \alpha) \xot{p_1} (Q, \chi) \xto{p_2} (B, \beta)
$$

satisfying $fp_1 = gp_2$, hence also $W(f) \circ \alpha p_1 = W(g) \circ
\beta p_2$, there is a unique factorisation

$$
\langle \alpha p_1 , \beta p_2 \rangle : Q \to WP\text{,}
$$

which we can compose with the counit $\eps$ to give a morphism $\nu : Q
\to P$ satisfying $\pi_1\nu = p_1$ and $\pi_2\nu = p_2$, as shown in
the formalisation below.

```agda
      abstract
        wit' : W₁ (f .hom) ∘ α ∘ p1 .hom ≡ W₁ (g .hom) ∘ β ∘ p2 .hom
        wit' = W₁ (f .hom) ∘ α ∘ p1 .hom   ≡⟨ pulll (f .preserves) ⟩
               (ξ ∘ f .hom) ∘ p1 .hom      ≡⟨ pullr wit ⟩
               ξ ∘ g .hom ∘ p2 .hom        ≡⟨ extendl (sym (g .preserves)) ⟩
               W₁ (g .hom) ∘ β ∘ p2 .hom   ∎

      ν : Hom Q P
      ν = counit.ε _ ∘ p'.universal wit'

      abstract
        comm₁ : π₁ .hom ∘ ν ≡ p1 .hom
        comm₁ =
          π₁ .hom ∘ ν                                    ≡⟨ pulll (sym (counit.is-natural _ _ _)) ⟩
          (counit.ε _ ∘ W₁ (π₁ .hom)) ∘ p'.universal _  ≡⟨ pullr p'.p₁∘universal ⟩
          counit.ε _ ∘ α ∘ p1 .hom                       ≡⟨ cancell α.ρ-counit ⟩
          p1 .hom                                        ∎
```

Since equality of morphisms in $\cC_W$ is entirely tested downstairs in
$\cC$, if $\rho$ extends to a coalgebra homomorphism, then it is
precisely the factorisation we're seeking for the square to be a
pullback in $\cC_W$. That's because, since it's induced by the universal
property of $P$ as a pullback, it automatically satisfies the
commutativity and uniqueness requirements.

<!--
```agda
        comm₂ : π₂ .hom ∘ ν ≡ p2 .hom
        comm₂ = pulll (sym (counit.is-natural _ _ _))
             ·· pullr p'.p₂∘universal
             ·· cancell β.ρ-counit
```
-->

It therefore remains to show that $W(\eps \langle \alpha p_1, \beta p_2
\rangle)\circ \chi$ is $\pi \eps \langle \alpha p_1, \beta p_2 \rangle$.
Since these are both maps into a pullback, namely $WP$, we can do so
componentwise. Therefore, we calculate

```agda
      step₁ : W₁ (π₁ .hom) ∘ W₁ ν ∘ χ ≡ α ∘ p1 .hom
      step₁ =
        W₁ (π₁ .hom) ∘ W₁ ν ∘ χ                    ≡⟨ pulll (W.weave (pulll (sym (counit.is-natural _ _ _)) ∙ pullr p'.p₁∘universal)) ⟩
        (W₁ (counit.ε _) ∘ W₁ (α ∘ p1 .hom)) ∘ χ   ≡⟨ pullr (ap₂ _∘_ (W-∘ _ _) refl ∙ pullr (p1 .preserves)) ⟩
        W₁ (counit.ε _) ∘ W₁ α ∘ α ∘ p1 .hom       ≡⟨ W.cancell α.ρ-counit ⟩
        α ∘ p1 .hom                                ∎
```

and

```agda
      step₂ : W₁ (π₁ .hom) ∘ π ∘ ν ≡ α ∘ p1 .hom
      step₂ = W₁ (π₁ .hom) ∘ π ∘ ν ≡⟨ pulll (π₁ .preserves) ⟩
              (α ∘ π₁ .hom) ∘ ν    ≡⟨ pullr comm₁ ⟩
              α ∘ p1 .hom          ∎
```

with very similar, but symmetric, proofs for the second projection ---
so $\nu$ was indeed the map we were looking for!

<!--
```agda
      factor : Coalgebras.Hom W (Q , χ') (P , π')
      factor .hom       = ν
      factor .preserves = p'.unique₂ {p = wit'} step₁
        (  pulll (W.weave (pulll (sym (counit.is-natural _ _ _)) ∙ pullr p'.p₂∘universal))
        ·· pullr (ap₂ _∘_ (W-∘ _ _) refl ∙ pullr (p2 .preserves))
        ·· W.cancell β.ρ-counit)
        step₂
        (pulll (π₂ .preserves) ∙ pullr comm₂)

```
-->

```agda
    pb' : is-pullback (Coalgebras W) π₁ f π₂ g
    pb' .square = ext p.square
    pb' .universal {p₁' = p₁'} {p₂'} x = factor p₁' p₂' (ap hom x)
    pb' .p₁∘universal {p₁' = p₁'} {p₂'} {p} = ext $ comm₁ p₁' p₂' (ap hom p)
    pb' .p₂∘universal {p₁' = p₁'} {p₂'} {p} = ext $ comm₂ p₁' p₂' (ap hom p)
    pb' .unique {p₁' = p₁'} {p₂'} {p} q r = ext $ p.unique₂
      {p = ap hom p} (ap hom q) (ap hom r)
      (comm₁ p₁' p₂' (ap hom p)) (comm₂ p₁' p₂' (ap hom p))
```

```agda
  Coalgebra-on-pullback
    : (p : Pullback C (f .hom) (g .hom))
    → Coalgebra-on W (Pullback.apex p)
  Coalgebra-on-pullback p = coalg where
```

<!--
```agda
    rem₁ = pres-pullback (p .has-is-pb)
    rem₂ = pres-pullback rem₁
    module p' = is-pullback rem₁
    module p = Pullback p
```
-->

Suppose now that we have a pullback

~~~{.quiver .short-05}
\[\begin{tikzcd}[ampersand replacement=\&]
  {A \times_X B} \&\& B \\
  \\
  A \&\& {X\text{,}}
  \arrow["f", from=1-3, to=3-3]
  \arrow["g"', from=3-1, to=3-3]
  \arrow["{p_1}"', from=1-1, to=3-1]
  \arrow["{p_2}", from=1-1, to=1-3]
  \arrow["\lrcorner"{anchor=center, pos=0.125}, draw=none, from=1-1, to=3-3]
\end{tikzcd}\]
~~~

and we want to equip $A \times_X B$ with a $W$-coalgebra structure. We
must come up with a map $A \times_X B \to W(A \times_X B)$, but since
$W$ preserves pullbacks, it will suffice to come up with a map $A
\times_X B \to WA \times_{WX} WB$. Since that is a pullback, we can
write such a map as $\langle x , y \rangle$ given $x : A \times_X B \to
WA$ and $y : A \times_X B \to WB$, as long as we have $W(f)x = W(g)y$.

We define $x$ (resp. $y$) to be the composite $A \times_X B \xto{p_1} A
\xto{\alpha} WA$ (resp. $p_2$, $\beta$), and a short calculation shows
that this indeed satisfies the aforementioned compatibility condition.

```agda
    abstract
      wit : W₁ (f .hom) ∘ α ∘ p.p₁ ≡ W₁ (g .hom) ∘ β ∘ p.p₂
      wit = W₁ (f .hom) ∘ α ∘ p.p₁  ≡⟨ pulll (f .preserves) ⟩
            (ξ ∘ f .hom) ∘ p.p₁     ≡⟨ pullr p.square ⟩
            ξ ∘ g .hom ∘ p.p₂       ≡⟨ extendl (sym (g .preserves)) ⟩
            W₁ (g .hom) ∘ β ∘ p.p₂  ∎

    coalg : Coalgebra-on W p.apex
    coalg .ρ = p'.universal wit
```

It remains to show that this factorisation is compatible with $W$'s
counit and comultiplication. We start with the former: we must show that
$\eps \langle \alpha p_1 , \beta p_2 \rangle$ is the identity. Since
equality of maps into pullbacks is determined by their composition with
the projections, it will suffice to show that

$$
p_1 \eps \langle \alpha p_1 , \beta p_2 \rangle
$$

is $p_1$, and respectively for $p_2$, but we can calculate

$$
p_1 \eps \langle \alpha p_1 , \beta p_2 \rangle
= \eps p_1 \langle \alpha p_1 , \beta p_2 \rangle
= \eps \alpha p_1
= p_1
$$

where the first step is by naturality, the second is computation, and
the third is because $\alpha$ is a $W$-coalgebra. To show compatibility
with the comultiplication, we perform a similar calculation: it will
suffice to show that

$$
WW(p_1) \circ W\langle \alpha p_1 , \beta p_2 \rangle \circ \langle \alpha p_1 , \beta p_2 \rangle
$$

and

$$
WW(p_1) \circ \delta \circ \langle \alpha p_1 , \beta p_2 \rangle
$$

are both $W(\alpha) \circ \alpha p_1$. In the former case this is
immediate, but in the latter case it requires naturality of $\delta$ and
compatibility of $\alpha$ with the comultiplication:

$$
\begin{align*}
& \phantom{=} WW(p_1) \circ \delta \circ \langle \alpha p_1 , \beta p_2 \rangle \\
& = \delta \circ W(p_1) \circ \langle \alpha p_1 , \beta p_2 \rangle            \\
& = \delta \circ \alpha p_1                                                     \\
& = W(\alpha) \circ \alpha p_1 \text{.}
\end{align*}
$$

<details>
<summary>Since putting the above paragraphs together into an actual
formalised proof is a bit annoying, it's hidden in this
`<details>`{.html} tag.</summary>

```agda
    coalg .ρ-counit =
      p.unique₂
        {p = pulll (sym (counit.is-natural _ _ _))
          ·· pullr wit
          ·· extendl (counit.is-natural _ _ _)}
        (pulll (sym (counit.is-natural _ _ _)) ∙ pullr p'.p₁∘universal)
        (pulll (sym (counit.is-natural _ _ _)) ∙ pullr p'.p₂∘universal)
        (idr _ ∙ sym (cancell α.ρ-counit))
        (idr _ ∙ sym (cancell β.ρ-counit))
    coalg .ρ-comult =
      is-pullback.unique₂ rem₂
        {p = W.extendl (f .preserves)
          ·· ap₂ _∘_ refl wit
          ·· W.extendl (sym (g .preserves))}
        (pulll (W.weave p'.p₁∘universal) ∙ pullr p'.p₁∘universal)
        (pulll (W.weave p'.p₂∘universal) ∙ pullr p'.p₂∘universal)
        (pulll (sym (comult.is-natural _ _ _)) ·· pullr p'.p₁∘universal ·· extendl (sym α.ρ-comult))
        (  pulll (sym (comult.is-natural _ _ _))
        ·· pullr p'.p₂∘universal
        ·· extendl (sym β.ρ-comult))
```

</details>

```agda
Pullback-coalgebra
  : {(A , α) (B , β) (X , ξ) : Coalgebras.Ob W}
    (f : Coalgebras.Hom W (A , α) (X , ξ))
    (g : Coalgebras.Hom W (B , β) (X , ξ))
  → Pullback (Coalgebras W) f g
```

<!--
```agda
Pullback-coalgebra f g = pb' where
  pb = pullbacks (f .hom) (g .hom)
  rem₁ = pres-pullback (pb .has-is-pb)

  pb' : Pullback (Coalgebras W) _ _
  pb' .apex .fst = _
  pb' .apex .snd = Coalgebra-on-pullback f g pb

  pb' .p₁ .hom       = Pullback.p₁ pb
  pb' .p₁ .preserves = rem₁ .p₁∘universal

  pb' .p₂ .hom       = Pullback.p₂ pb
  pb' .p₂ .preserves = rem₁ .p₂∘universal

  pb' .has-is-pb = is-pullback-coalgebra f g (pb .has-is-pb)

open Terminal
```
-->

The case of the terminal object is infinitely simpler: since $W$
preserves the terminal object, we have $W\top \cong \top$. We equip it
with its (necessarily unique) cofree coalgebra structure, and since
there is an isomorphism

$$
\hom_{\cC_W}((A, \alpha), W\top) \cong \hom_{\cC}(A, \top)
$$

the former is contractible if the latter is.

```agda
Terminal-coalgebra : Terminal (Coalgebras W)
Terminal-coalgebra .top = _
Terminal-coalgebra .has⊤ (A , α) = Equiv→is-hlevel 0
  (Equiv.inverse (_ , L-adjunct-is-equiv (Forget⊣Cofree W)))
  (terminal .has⊤ A)
```

Since we have a terminal object and pullbacks, we have arbitrary finite
limits, too.

```agda
Coalgebras-finitely-complete : Finitely-complete (Coalgebras W)
Coalgebras-finitely-complete = with-pullbacks (Coalgebras W)
  Terminal-coalgebra
  Pullback-coalgebra
```
