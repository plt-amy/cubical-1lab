<!--
```agda
open import Cat.Diagram.Pullback.Properties
open import Cat.Diagram.Pullback
open import Cat.Prelude

import Cat.Displayed.Instances.Subobjects as Subobjs
import Cat.Reasoning as Cat
```
-->

```agda
module Cat.Diagram.Pullback.Along where
```

<!--
```agda
```
-->

# Pullbacks along an indeterminate morphism

<!--
```agda
module _ {o ℓ} (C : Precategory o ℓ) where
  open Precategory C
```
-->

This module defines the auxiliary notion of a map $p_1 : P \to X$ being
the [[pullback]] of a map $g : Y \to Z$ along a map $f : X \to Z$, by
summing over the possible maps $P \to Y$ fitting in the diagram below.

~~~{.quiver}
\[\begin{tikzcd}[ampersand replacement=\&]
  P \&\& Y \\
  \\
  X \&\& Z
  \arrow[dashed, from=1-1, to=1-3]
  \arrow["{p_1}"', from=1-1, to=3-1]
  \arrow["\lrcorner"{anchor=center, pos=0.125}, draw=none, from=1-1, to=3-3]
  \arrow["g", from=1-3, to=3-3]
  \arrow["f"', from=3-1, to=3-3]
\end{tikzcd}\]
~~~

While this is a rather artificial notion, it comes up a lot in the
context of [[elementary topos]] theory, especially in the case where $g$
is the [[generic subobject]] $\true : \top \to \Omega$.

```agda
  record is-pullback-along {P X Y Z} (p₁ : Hom P X) (f : Hom X Z) (g : Hom Y Z) : Type (o ⊔ ℓ) where
    no-eta-equality
    field
      top       : Hom P Y
      has-is-pb : is-pullback C p₁ f top g
    open is-pullback has-is-pb public
```

<!--
```
open is-pullback-along
open is-pullback
open Pullback

module _ {o ℓ} {C : Precategory o ℓ} where
  open Subobjs C
  open Cat C

  private
    unquoteDecl eqv = declare-record-iso eqv (quote is-pullback-along)
    variable
      U V W X Y Z : Ob
      f g h k m n l nm : Hom X Y
  abstract
```
-->

::: warning
Despite the name `is-pullback-along`{.Agda}, nothing about the
definition guarantees that this type is a [[proposition]] after fixing
the three maps $p_1$, $f$ and $g$. However, we choose to name this
auxiliary notion as though it were always a proposition, since it will
most commonly be used when $g$ is a [[monomorphism]], and, under this
assumption, it *is* propositional:
:::

```agda
    is-monic→is-pullback-along-is-prop
      : ∀ {P X Y Z} {f : Hom P X} {g : Hom X Z} {h : Hom Y Z}
      → is-monic h
      → is-prop (is-pullback-along C f g h)
    is-monic→is-pullback-along-is-prop h-monic = Iso→is-hlevel 1 eqv λ (_ , x) (_ , y) →
      Σ-prop-path! (h-monic _ _ (sym (x .square) ∙ y .square))
```

Of course, the notion of being a pullback *along* a map is closed under
composition: if $k$ is the pullback of $l$ along $n$, and $h$ is the
pullback of $k$ along $m$, as in the diagram below, then $h$ is also the
pullback of $l$ along $n \circ m$.

~~~{.quiver}
\[\begin{tikzcd}[ampersand replacement=\&]
  U \&\& V \&\& W \\
  \\
  X \&\& Y \&\& Z
  \arrow[dashed, from=1-1, to=1-3]
  \arrow["h"', from=1-1, to=3-1]
  \arrow["\lrcorner"{anchor=center, pos=0.125}, draw=none, from=1-1, to=3-3]
  \arrow[dashed, from=1-3, to=1-5]
  \arrow["k"{description}, from=1-3, to=3-3]
  \arrow["\lrcorner"{anchor=center, pos=0.125}, draw=none, from=1-3, to=3-5]
  \arrow["l", from=1-5, to=3-5]
  \arrow["m"', from=3-1, to=3-3]
  \arrow["n"', from=3-3, to=3-5]
\end{tikzcd}\]
~~~

```agda
  paste-is-pullback-along
    : is-pullback-along C k n l → is-pullback-along C h m k
    → n ∘ m ≡ nm → is-pullback-along C h nm l
  paste-is-pullback-along p q r .top = p .top ∘ q .top
  paste-is-pullback-along p q r .has-is-pb =
    subst-is-pullback refl r refl refl (rotate-pullback (pasting-left→outer-is-pullback
      (rotate-pullback (p .has-is-pb))
      (rotate-pullback (q .has-is-pb))
      ( extendl (sym (p .is-pullback-along.square))
      ∙ pushr (sym (q .is-pullback-along.square)))))
```
