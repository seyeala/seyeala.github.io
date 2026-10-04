'use strict';

document.documentElement.classList.add('js');
const menuButton = document.querySelector('.nav-toggle');
const navigation = document.getElementById('primary-navigation');
if (menuButton && navigation) {
  navigation.dataset.collapsed = 'true';
  menuButton.hidden = false;
  menuButton.addEventListener('click', () => {
    const open = menuButton.getAttribute('aria-expanded') !== 'true';
    menuButton.setAttribute('aria-expanded', String(open));
    navigation.dataset.collapsed = String(!open);
  });
  document.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    const expanded = menuButton.getAttribute('aria-expanded') === 'true';
    document.querySelectorAll('.more-nav[open]').forEach(menu => { menu.open = false; });
    if (expanded) {
      menuButton.setAttribute('aria-expanded', 'false');
      navigation.dataset.collapsed = 'true';
      menuButton.focus();
    }
  });
}

document.querySelectorAll('.member-photo img').forEach(image => {
  function showFallback() {
    image.hidden = true;
    const frame = image.parentElement;
    frame.setAttribute('role', 'img');
    frame.setAttribute('aria-label', frame.dataset.fallbackLabel);
  }
  image.addEventListener('error', showFallback);
  if (image.complete && image.naturalWidth === 0) showFallback();
});
