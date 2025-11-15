// Cookie utility functions for trial tracking

export const getCookie = (name) => {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(';').shift();
  return null;
};

export const setCookie = (name, value, days = 365) => {
  const date = new Date();
  date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
  const expires = `expires=${date.toUTCString()}`;
  document.cookie = `${name}=${value};${expires};path=/;SameSite=Lax`;
};

export const getOrCreateTrialCookie = () => {
  let cookieId = getCookie('lipika_trial_id');
  if (!cookieId) {
    // Generate a simple ID (in production, use UUID)
    cookieId = `trial_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setCookie('lipika_trial_id', cookieId);
  }
  return cookieId;
};

