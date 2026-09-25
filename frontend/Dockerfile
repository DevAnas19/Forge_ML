FROM node:20-alpine AS base

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# Next.js needs to know the backend URL AT BUILD TIME for anything using
# NEXT_PUBLIC_* env vars in Server Components, since those get baked into
# the build output -- not read fresh at container startup like you might
# expect from a typical Node app. We'll pass this in via docker-compose.
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]