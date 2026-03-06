// Mock product data for the search gallery
// In production, you'd crop the segmented region and send it to a visual search API

export interface Product {
  id: string;
  name: string;
  price: string;
  image: string;
  brand: string;
}

const PLACEHOLDER = "https://picsum.photos/seed";

// Generate deterministic mock products based on the clicked region
export function getMockProducts(seedHash: number): Product[] {
  const categories = [
    { names: ["Classic Aviator Sunglasses", "Retro Round Frames", "Sport Shield Sunglasses", "Cat-Eye Glasses", "Oversized Square Frames", "Wayfarer Style Glasses"], brands: ["Ray-Ban", "Oakley", "Gucci", "Prada", "Tom Ford", "Persol"] },
    { names: ["Slim Fit Oxford Shirt", "Oversized Graphic Tee", "Linen Button-Down", "Cropped Tank Top", "Striped Polo Shirt", "Silk Blouse"], brands: ["Uniqlo", "Zara", "H&M", "COS", "& Other Stories", "Everlane"] },
    { names: ["High-Waist Straight Jeans", "Pleated Wide-Leg Trousers", "Cargo Joggers", "Tailored Chinos", "Corduroy Pants", "Leather Leggings"], brands: ["Levi's", "Acne Studios", "Citizens of Humanity", "AG Jeans", "Frame", "Agolde"] },
    { names: ["Chunky Leather Sneakers", "Chelsea Boots", "Suede Loafers", "Running Shoes", "Platform Sandals", "Canvas High-Tops"], brands: ["Nike", "Adidas", "New Balance", "Common Projects", "Dr. Martens", "Veja"] },
    { names: ["Quilted Puffer Jacket", "Double-Breasted Blazer", "Denim Trucker Jacket", "Trench Coat", "Leather Biker Jacket", "Wool Overcoat"], brands: ["North Face", "AllSaints", "Barbour", "Burberry", "Acne Studios", "Max Mara"] },
    { names: ["Crossbody Mini Bag", "Canvas Tote", "Leather Backpack", "Belt Bag", "Structured Satchel", "Woven Clutch"], brands: ["Coach", "Telfar", "Mansur Gavriel", "Loewe", "Bottega Veneta", "Celine"] },
  ];

  const catIdx = Math.abs(seedHash) % categories.length;
  const cat = categories[catIdx];

  return cat.names.map((name, i) => ({
    id: `${catIdx}-${i}`,
    name,
    price: `$${(Math.abs((seedHash * (i + 1) * 17) % 400) + 29).toFixed(0)}`,
    image: `${PLACEHOLDER}/${Math.abs(seedHash + i * 31)}/200/260`,
    brand: cat.brands[i],
  }));
}

export function hashPoint(x: number, y: number): number {
  return Math.floor(x * 1000 + y * 7919);
}
