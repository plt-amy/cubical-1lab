<!--
```agda
open import Cat.Displayed.Cartesian
open import Cat.Displayed.Total.Op
open import Cat.Displayed.Base
open import Cat.Prelude

import Cat.Displayed.Morphism.Duality
import Cat.Displayed.Reasoning as DR
import Cat.Displayed.Morphism
import Cat.Reasoning
```
-->

```agda
module Cat.Displayed.Cocartesian
  {o ℓ o' ℓ'} {ℬ : Precategory o ℓ} (ℰ : Displayed ℬ o' ℓ') where

open Cat.Reasoning ℬ
open Displayed ℰ
open Cat.Displayed.Morphism ℰ
open Cat.Displayed.Morphism.Duality ℰ
open DR ℰ
```

# Cocartesian morphisms and opfibrations

[[Cartesian fibrations]] provide a way of describing pseudofunctorial
families of categories $\ca{B}^{op} \to \Cat$ purely in terms of
displayed structure. It's then natural to ask: what about
*covariant* pseudofunctorial families of categories? Such pseudofunctors
can be encoded by dualising cartesian fibrations.

:::{.definition #cocartesian-morphism alias="cocartesian-map"}
To do this, we must first dualise the notion of a cartesian map to a
**cocartesian map**. Fix a map $a \to b$ in $\ca{B}$, objects $a'$
and $b'$ displayed over $a$ and $b$ resp., and a map $f' : a' \to_f b'$
over $f$. We say that $f'$ is **cocartesian** if it has the shape of a
"pushout diagram", in contrast to the "pullback diagrams" shape
associated with cartesian maps.
:::

~~~{.quiver}
\[\begin{tikzcd}
  {a'} && {b'} \\
  \\
  a && b
  \arrow["{f'}"', from=1-1, to=1-3]
  \arrow["f", from=3-1, to=3-3]
  \arrow[lies over, from=1-1, to=3-1]
  \arrow[lies over, from=1-3, to=3-3]
  \arrow["\lrcorner"{anchor=center, pos=0.125, rotate=180}, draw=none, from=3-3, to=1-1]
\end{tikzcd}\]
~~~

```agda
record is-cocartesian
  {a b a' b'} (f : Hom a b) (f' : Hom[ f ] a' b')
  : Type (o ⊔ ℓ ⊔ o' ⊔ ℓ')
  where
  no-eta-equality
```

Concretely, let $u : \ca{B}$ and $u'$ be displayed over $u$. Furthermore,
suppose we have a map $m : b \to u$ (highlighted in violet below), along
with a map $h' : a' \to_{mf} u'$ displayed over $m \cdot f$ (highlighted
in red). For $f'$ to be cocartesian, every such situation must give rise
to a unique universal factorisation of $h'$ through a map $b' \to_{m} u'$
(highlighted in green).

~~~{.quiver}
\begin{tikzcd}
	&&& \textcolor{rgb,255:red,124;green,50;blue,189}{u'} \\
	{a'} && {b'} \\
	&&& \textcolor{rgb,255:red,124;green,50;blue,189}{u} \\
	a && b
	\arrow[lies over, from=2-1, to=4-1]
	\arrow["f", from=4-1, to=4-3]
	\arrow[lies over, from=2-3, to=4-3]
	\arrow["{f'}", from=2-1, to=2-3]
	\arrow["{m'}", color={rgb,255:red,153;green,92;blue,214}, from=4-3, to=3-4]
	\arrow["{\exists!}"', color={rgb,255:red,92;green,214;blue,92}, dashed, from=2-3, to=1-4]
	\arrow[lies over, color={rgb,255:red,153;green,92;blue,214}, from=1-4, to=3-4]
	\arrow["{h'}", color={rgb,255:red,214;green,92;blue,92}, curve={height=-12pt}, from=2-1, to=1-4]
\end{tikzcd}
~~~

```agda
  field
    universal
      : ∀ {u u'} (m : Hom b u) (h' : Hom[ m ∘ f ] a' u')
      → Hom[ m ] b' u'
    commutes
      : ∀ {u u'} (m : Hom b u) (h' : Hom[ m ∘ f ] a' u')
      → universal m h' ∘' f' ≡ h'
    unique
      : ∀ {u u'} {m : Hom b u} {h' : Hom[ m ∘ f ] a' u'}
      → (m' : Hom[ m ] b' u') → m' ∘' f' ≡ h'
      → m' ≡ universal m h'
```

<!--
```agda
  universal' : ∀ {u u'} {m : Hom b u} {k : Hom a u}
             → (p : m ∘ f ≡ k) (h' : Hom[ k ] a' u')
             → Hom[ m ] b' u'
  universal' {u' = u'} p h' = universal _ (coe1→0 (λ i → Hom[ p i ] a' u') h')

  commutesp : ∀ {u u'} {m : Hom b u} {k : Hom a u}
            → (p : m ∘ f ≡ k) (h' : Hom[ k ] a' u')
            → universal' p h' ∘' f' ≡[ p ] h'
  commutesp {u' = u'} p h' =
    to-pathp⁻ (commutes _ (coe1→0 (λ i → Hom[ p i ] a' u') h'))

  universalp : ∀ {u u'} {m₁ m₂ : Hom b u} {k : Hom a u}
             → (p : m₁ ∘ f ≡ k) (q : m₁ ≡ m₂) (r : m₂ ∘ f ≡ k)
             → (h' : Hom[ k ] a' u')
             → universal' p h' ≡[ q ] universal' r h'
  universalp {u = u} p q r h' i =
    universal' (is-set→squarep (λ _ _ → Hom-set a u) (ap (_∘ f) q) p r refl i) h'

  uniquep : ∀ {u u'} {m₁ m₂ : Hom b u} {k : Hom a u}
          → (p : m₁ ∘ f ≡ k) (q : m₁ ≡ m₂) (r : m₂ ∘ f ≡ k)
          → {h' : Hom[ k ] a' u'}
          → (m' : Hom[ m₁ ] b' u')
          → m' ∘' f' ≡[ p ] h' → m' ≡[ q ] universal' r h'
  uniquep p q r {h' = h'} m' s  =
    to-pathp⁻ (unique m' (from-pathp⁻ s) ∙ from-pathp⁻ (universalp p q r h'))

  uniquep₂ : ∀ {u u'} {m₁ m₂ : Hom b u} {k : Hom a u}
          → (p : m₁ ∘ f ≡ k) (q : m₁ ≡ m₂) (r : m₂ ∘ f ≡ k)
          → {h' : Hom[ k ] a' u'}
          → (m₁' : Hom[ m₁ ] b' u')
          → (m₂' : Hom[ m₂ ] b' u')
          → m₁' ∘' f' ≡[ p ] h'
          → m₂' ∘' f' ≡[ r ] h'
          → m₁' ≡[ q ] m₂'
  uniquep₂ p q r {h' = h'} m₁' m₂' α β = to-pathp⁻ $
       unique m₁' (from-pathp⁻ α)
    ·· from-pathp⁻ (universalp p q r _)
    ·· ap hom[] (sym (unique m₂' (from-pathp⁻ β)))

  universalv : ∀ {b''} (f'' : Hom[ f ] a' b'') → Hom[ id ] b' b''
  universalv f'' = universal' (idl _) f''

  commutesv
    : ∀ {x'} (g' : Hom[ f ] a' x')
    → universalv g' ∘' f' ≡[ idl _ ] g'
  commutesv = commutesp (idl _)

  uniquev
    : ∀ {x'} {g' : Hom[ f ] a' x'}
    → (h' : Hom[ id ] b' x')
    → h' ∘' f' ≡[ idl _ ] g'
    → h' ≡ universalv g'
  uniquev h' p = uniquep (idl _) refl (idl _) h' p

  uniquev₂
    : ∀ {x'} {g' : Hom[ f ] a' x'}
    → (h' h'' : Hom[ id ] b' x')
    → h' ∘' f' ≡[ idl _ ] g'
    → h'' ∘' f' ≡[ idl _ ] g'
    → h' ≡ h''
  uniquev₂ h' h'' p q = uniquep₂ (idl _) refl (idl _) h' h'' p q

record Cocartesian-morphism
  {x y : Ob} (f : Hom x y) (x' : Ob[ x ]) (y' : Ob[ y ])
  : Type (o ⊔ ℓ ⊔ o' ⊔ ℓ') where
  no-eta-equality
  field
    hom' : Hom[ f ] x' y'
    cocartesian : is-cocartesian f hom'

  open is-cocartesian cocartesian public
```
-->

## Duality

As noted before, cocartesian maps the are duals to [[cartesian maps]].
We can make this correspondence precise by showing that cartesian maps
in the [total opposite] of $\ca{E}$ are cocartesian maps, and vice
versa.

[total opposite]: Cat.Displayed.Total.Op.html

```agda
co-cartesian→cocartesian
  : ∀ {x y} {f : Hom x y} {x' y'} {f' : Hom[ f ] x' y'}
  → is-cartesian (ℰ ^total-op) f f'
  → is-cocartesian f f'

cocartesian→co-cartesian
  : ∀ {x y} {f : Hom x y} {x' y'} {f' : Hom[ f ] x' y'}
  → is-cocartesian f f'
  → is-cartesian (ℰ ^total-op) f f'
```

<details>
<summary>These functions just unpack and repack data, they are not
particularly interesting.
</summary>

```agda
co-cartesian→cocartesian cart^op .is-cocartesian.universal m h' =
  is-cartesian.universal cart^op m h'
co-cartesian→cocartesian cart^op .is-cocartesian.commutes m h' =
  is-cartesian.commutes cart^op m h'
co-cartesian→cocartesian cart^op .is-cocartesian.unique m' p =
  is-cartesian.unique cart^op m' p

cocartesian→co-cartesian cocart .is-cartesian.universal m h' =
  is-cocartesian.universal cocart m h'
cocartesian→co-cartesian cocart .is-cartesian.commutes m h' =
  is-cocartesian.commutes cocart m h'
cocartesian→co-cartesian cocart .is-cartesian.unique m' p =
  is-cocartesian.unique cocart m' p
```
</details>

Furthermore, these 2 functions form an equivalence, which we can extend
to a path via univalence.

```agda
co-cartesian→cocartesian-is-equiv
  : ∀ {x y} {f : Hom x y} {x' y'} {f' : Hom[ f ] x' y'}
  → is-equiv (co-cartesian→cocartesian {f' = f'})
co-cartesian→cocartesian-is-equiv {f' = f'} =
  is-iso→is-equiv $ iso cocartesian→co-cartesian cocart-invl cocart-invr
  where
    cocart-invl
      : ∀ f
      → co-cartesian→cocartesian {f' = f'} (cocartesian→co-cartesian f) ≡ f
    cocart-invl f i .is-cocartesian.universal m h' = is-cocartesian.universal f m h'
    cocart-invl f i .is-cocartesian.commutes m h' = is-cocartesian.commutes f m h'
    cocart-invl f i .is-cocartesian.unique m' p = is-cocartesian.unique f m' p

    cocart-invr
      : ∀ f
      → cocartesian→co-cartesian {f' = f'} (co-cartesian→cocartesian f) ≡ f
    cocart-invr f i .is-cartesian.universal m h' = is-cartesian.universal f m h'
    cocart-invr f i .is-cartesian.commutes m h' = is-cartesian.commutes f m h'
    cocart-invr f i .is-cartesian.unique m' p = is-cartesian.unique f m' p

co-cartesian≡cocartesian
  : ∀ {x y} {f : Hom x y} {x' y'} {f' : Hom[ f ] x' y'}
  → is-cartesian (ℰ ^total-op) f f' ≡ is-cocartesian f f'
co-cartesian≡cocartesian =
  ua (co-cartesian→cocartesian , co-cartesian→cocartesian-is-equiv)
```

## Properties of cocartesian morphisms

We shall now prove the following properties of cocartesian morphisms.

```agda
cocartesian-∘
  : ∀ {x y z} {f : Hom y z} {g : Hom x y}
  → ∀ {x' y' z'} {f' : Hom[ f ] y' z'} {g' : Hom[ g ] x' y'}
  → is-cocartesian f f' → is-cocartesian g g'
  → is-cocartesian (f ∘ g) (f' ∘' g')

cocartesian-id : ∀ {x x'} → is-cocartesian id (id' {x} {x'})

invertible→cocartesian
  : ∀ {x y} {f : Hom x y} {x' y'} {f' : Hom[ f ] x' y'}
  → (f-inv : is-invertible f)
  → is-invertible[ f-inv ] f'
  → is-cocartesian f f'

cocartesian→weak-epic
  : ∀ {x y} {f : Hom x y}
  → ∀ {x' y'} {f' : Hom[ f ] x' y'}
  → is-cocartesian f f'
  → is-weak-epic f'

cocartesian-codomain-unique
  : ∀ {x y} {f : Hom x y}
  → ∀ {x' y' y''} {f' : Hom[ f ] x' y'} {f'' : Hom[ f ] x' y''}
  → is-cocartesian f f'
  → is-cocartesian f f''
  → y' ≅↓ y''

cocartesian-vertical-section-stable
  : ∀ {x y} {f : Hom x y}
  → ∀ {x' y' y''} {f' : Hom[ f ] x' y'} {f'' : Hom[ f ] x' y''} {ϕ : Hom[ id ] y'' y'}
  → is-cocartesian f f'
  → has-retract↓ ϕ
  → ϕ ∘' f'' ≡[ idl _ ] f'
  → is-cocartesian f f''

cocartesian-pasting
  : ∀ {x y z} {f : Hom y z} {g : Hom x y}
  → ∀ {x' y' z'} {f' : Hom[ f ] y' z'} {g' : Hom[ g ] x' y'}
  → is-cocartesian g g'
  → is-cocartesian (f ∘ g) (f' ∘' g')
  → is-cocartesian f f'

vertical+cocartesian→invertible
  : ∀ {x} {x' x'' : Ob[ x ]} {f' : Hom[ id ] x' x''}
  → is-cocartesian id f'
  → is-invertible↓ f'
```

<details>
<summary>The proofs are all applications of duality, as we have already
done the hard work of proving these properties for [[cartesian
morphisms]].
</summary>

```agda
cocartesian-∘ f-cocart g-cocart =
  co-cartesian→cocartesian $
  cartesian-∘ _
    (cocartesian→co-cartesian g-cocart)
    (cocartesian→co-cartesian f-cocart)

cocartesian-id = co-cartesian→cocartesian (cartesian-id _)

invertible→cocartesian f-inv f'-inv =
  co-cartesian→cocartesian $
  invertible→cartesian _ _ (invertible[]→co-invertible[] f'-inv)

cocartesian→weak-epic cocart =

  cartesian→weak-monic (ℰ ^total-op) (cocartesian→co-cartesian cocart)

cocartesian-codomain-unique f'-cocart f''-cocart =
  vertical-co-iso→vertical-iso $
  cartesian-domain-unique (ℰ ^total-op)
    (cocartesian→co-cartesian f''-cocart)
    (cocartesian→co-cartesian f'-cocart)

cocartesian-vertical-section-stable cocart ret factor =
  co-cartesian→cocartesian $
  cartesian-vertical-retraction-stable (ℰ ^total-op)
    (cocartesian→co-cartesian cocart)
    (vertical-retract→vertical-co-section ret)
    factor

cocartesian-pasting g-cocart fg-cocart =
  co-cartesian→cocartesian $
  cartesian-pasting (ℰ ^total-op)
    (cocartesian→co-cartesian g-cocart)
    (cocartesian→co-cartesian fg-cocart)

vertical+cocartesian→invertible cocart =
  vertical-co-invertible→vertical-invertible $
  vertical+cartesian→invertible (ℰ ^total-op)
    (cocartesian→co-cartesian cocart)
```
</details>

Furthermore, $f' : x' \to_{f} y'$ is cocartesian if and only if the
function $- \cdot' f$ is an equivalence.

```agda
precompose-equiv→cocartesian
  : ∀ {x y x' y'} {f : Hom x y}
  → (f' : Hom[ f ] x' y')
  → (∀ {z z'} {g : Hom y z} → is-equiv {A = Hom[ g ] y' z'} (_∘' f'))
  → is-cocartesian f f'
precompose-equiv→cocartesian f' cocart =
  co-cartesian→cocartesian $
  postcompose-equiv→cartesian (ℰ ^total-op) f' cocart

cocartesian→precompose-equiv
  : ∀ {x y z x' y' z'} {g : Hom y z} {f : Hom x y} {f' : Hom[ f ] x' y'}
  → is-cocartesian f f'
  → is-equiv {A = Hom[ g ] y' z'} (_∘' f')
cocartesian→precompose-equiv cocart =
  cartesian→postcompose-equiv (ℰ ^total-op) $
  cocartesian→co-cartesian cocart
```


## Cocartesian lifts {defines="cocartesian-lift"}

We call an object $b'$ over $b$ together with a cartesian arrow
$f' : a \to_{f} b'$ a **cocartesian lift** of $f$.

```agda
record Cocartesian-lift {x y} (f : Hom x y) (x' : Ob[ x ]) : Type (o ⊔ ℓ ⊔ o' ⊔ ℓ')
  where
  no-eta-equality
  field
    {y'} : Ob[ y ]
    lifting : Hom[ f ] x' y'
    cocartesian : is-cocartesian f lifting
  open is-cocartesian cocartesian public
```

We also can apply duality to cocartesian lifts.

```agda
co-cartesian-lift→cocartesian-lift
  : ∀ {x y} {f : Hom x y} {x' : Ob[ x ]}
  → Cartesian-lift (ℰ ^total-op) f x'
  → Cocartesian-lift f x'

cocartesian-lift→co-cartesian-lift
  : ∀ {x y} {f : Hom x y} {x' : Ob[ x ]}
  → Cocartesian-lift f x'
  → Cartesian-lift (ℰ ^total-op) f x'
```
<details>
<summary>The proofs are simply applying duality, so they are not
particularly interesting.
</summary>

```agda
co-cartesian-lift→cocartesian-lift cart .Cocartesian-lift.y' =
  Cartesian-lift.x' cart
co-cartesian-lift→cocartesian-lift cart .Cocartesian-lift.lifting =
  Cartesian-lift.lifting cart
co-cartesian-lift→cocartesian-lift cart .Cocartesian-lift.cocartesian =
  co-cartesian→cocartesian (Cartesian-lift.cartesian cart)

cocartesian-lift→co-cartesian-lift cocart .Cartesian-lift.x' =
  Cocartesian-lift.y' cocart
cocartesian-lift→co-cartesian-lift cocart .Cartesian-lift.lifting =
  Cocartesian-lift.lifting cocart
cocartesian-lift→co-cartesian-lift cocart .Cartesian-lift.cartesian =
  cocartesian→co-cartesian (Cocartesian-lift.cocartesian cocart)
```
</details>

:::{.definition #cocartesian-fibration}

We can use this notion to define cocartesian fibrations (sometimes
referred to as **opfibrations**).

```agda
record Cocartesian-fibration : Type (o ⊔ ℓ ⊔ o' ⊔ ℓ') where
  no-eta-equality
  field
    has-lift : ∀ {x y} (f : Hom x y) (x' : Ob[ x ]) → Cocartesian-lift f x'

  module has-lift {x y} (f : Hom x y) (x' : Ob[ x ]) =
    Cocartesian-lift (has-lift f x')
```

:::

<!--

```agda
  rebase : ∀ {x y x' x''} → (f : Hom x y)
           → Hom[ id ] x' x''
           → Hom[ id ] (has-lift.y' f x') (has-lift.y' f x'')
  rebase f vert =
    has-lift.universalv f _ (hom[ idr _ ] (has-lift.lifting f _ ∘' vert))
```
-->

As expected, opfibrations are dual to fibrations.
```agda
op-fibration→opfibration : Cartesian-fibration (ℰ ^total-op) → Cocartesian-fibration

opfibration→op-fibration : Cocartesian-fibration → Cartesian-fibration (ℰ ^total-op)
```

<details>
<summary> The proofs here are just further applications of duality, so
we omit them.
</summary>
```agda
op-fibration→opfibration fib .Cocartesian-fibration.has-lift f x' =
  co-cartesian-lift→cocartesian-lift (Cartesian-fibration.has-lift fib f x')

opfibration→op-fibration opfib .Cartesian-fibration.has-lift f y' =
  cocartesian-lift→co-cartesian-lift (Cocartesian-fibration.has-lift opfib f y')
```
</details>
