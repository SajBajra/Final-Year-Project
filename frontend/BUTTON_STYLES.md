# Global Button Styles Reference

This document provides a reference for the global button styles available in the project.

## Button Variants

### Primary Button
```jsx
<button className="btn-primary btn-md">Click Me</button>
```
- **Use case**: Main actions, CTAs
- **Style**: Solid primary blue background

### Secondary Button  
```jsx
<button className="btn-secondary btn-md">Click Me</button>
```
- **Use case**: Secondary actions
- **Style**: Solid orange background

### Outline Button
```jsx
<button className="btn-outline btn-md">Click Me</button>
```
- **Use case**: Less prominent actions (e.g., Login)
- **Style**: Transparent with primary border, hover shows light background

### Outline Filled Button
```jsx
<button className="btn-outline-filled btn-md">Click Me</button>
```
- **Use case**: Alternative style with hover fill
- **Style**: White background with primary border, fills on hover

### Ghost Button
```jsx
<button className="btn-ghost btn-md">Click Me</button>
```
- **Use case**: Subtle actions
- **Style**: Transparent, gray text, light hover

### Danger Button
```jsx
<button className="btn-danger btn-md">Delete</button>
```
- **Use case**: Destructive actions
- **Style**: Solid red background

### Success Button
```jsx
<button className="btn-success btn-md">Confirm</button>
```
- **Use case**: Positive confirmations
- **Style**: Solid green background

## Button Sizes

- **Small**: `btn-sm` - Compact buttons
- **Medium**: `btn-md` - Default size
- **Large**: `btn-lg` - Prominent buttons

## Icon Buttons

```jsx
<button className="btn-icon btn-primary">
  <FaIcon />
</button>
```

- `btn-icon` - Regular icon button
- `btn-icon-sm` - Small icon button
- `btn-icon-lg` - Large icon button

## Disabled State

```jsx
<button className="btn-primary btn-md btn-disabled">
  Disabled
</button>
```

## Usage Examples

### Link Button
```jsx
<Link to="/page" className="btn-primary btn-md">
  Go to Page
</Link>
```

### Full Width Button
```jsx
<button className="btn-primary btn-lg w-full">
  Submit
</button>
```

### Button with Icon
```jsx
<button className="btn-primary btn-md">
  <FaUser />
  <span>Profile</span>
</button>
```

### Loading Button
```jsx
<button 
  className={`btn-primary btn-lg ${loading ? 'btn-disabled' : ''}`}
  disabled={loading}
>
  {loading ? 'Loading...' : 'Submit'}
</button>
```

## Custom Combinations

You can combine button styles with other Tailwind classes:

```jsx
<button className="btn-primary btn-md shadow-lg">
  Enhanced Button
</button>
```

## Notes

- All buttons include `transition-all duration-200` for smooth animations
- Buttons automatically flex their content with `gap-2` for icon spacing
- Use `w-full` for full-width buttons
- Disabled state uses `opacity-50` and `pointer-events-none`
